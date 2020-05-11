#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:49:47 2020

@author: Team17
"""


"""
Load Package
"""
!pip install transformers
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from datetime import datetime
from pyspark.sql.functions import col, udf, datediff, to_date, lit
from pyspark.sql.types import DateType
import pyspark.sql.functions as F
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
import pickle
import os
import string
import re
import matplotlib.pyplot as plt
 
"""
Load sparknlp
"""
import sparknlp
spark = sparknlp.start() # for GPU training >> sparknlp.start(gpu=True)
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import pandas as pd

"""
Connect to S3
"""
import urllib
import urllib.parse

ACCESS_KEY = "AKIAYM5PPQ5XOOKQS75K"
SECRET_KEY = "YUYR4RcKZ4iSS4e4%2FuUkIRdWF6F9gKYcpdowMdFJ"
ENCODED_SECRET_KEY = urllib.parse.quote(SECRET_KEY, "")
AWS_BUCKET_NAME = "team17nlp"
MOUNT_NAME = "s3data"

"""
Load icu and note file
"""
# File location and type
file_location = "dbfs:/mnt/s3data/NOTEEVENTS.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_notes = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
  
# Load icustay
file_location = "dbfs:/mnt/s3data/ICUSTAYS.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_icu = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)  
  
"""
Get Valid Note
"""
# Keep useful columns - SUBJECT_ID, HADM_ID, CHARTDATE, CATEGORY, ISERROR and TEXT
df_notes_red = df_notes.select([c for c in df_notes.columns if c not in ['ROW_ID', 'CHARTTIME', 'STORETIME', 'DESCRIPTION', 'CGID']])


# Convert CHARTDATE to date
func =  udf (lambda x: datetime.strptime(x, '%Y-%m-%d'), DateType())
df_notes_red = df_notes_red.withColumn('CHARTDATE', func(col('CHARTDATE')))

# Select rows where note is not an error
df_notes_valid = df_notes_red.drop(df_notes_red.ISERROR == '1')

# drop the ISERROR column now
df_notes_valid = df_notes_valid.select([c for c in df_notes_valid.columns if c not in ['ISERROR']])

# drop rows with missing HADM_ID
df_notes_valid = df_notes_valid.na.drop(subset=["HADM_ID"])

# only keep notes in certain categories
note_categories = ['Consult', 'Discharge summary', 'ECG', 'Physician ', 'Echo', 'Radiology']
df_notes_valid = df_notes_valid[df_notes_valid['CATEGORY'].isin(note_categories)]



"""
Get ICU revisit
"""
# Drop uninformative columns
df_icu = df_icu.select([c for c in df_icu.columns if c not in ['DBSOURCE', 'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'FIRST_WARDID', 'LAST_WARDID','LOS']])

# Get next icu date
df_icu.registerTempTable("icu")
df_icu_new = spark.sql("SELECT *, LEAD(INTIME)  OVER (PARTITION BY SUBJECT_ID ORDER BY INTIME) AS NEXT_VISIT from icu")

# Add days until next ICU visit
df_icu_new = df_icu_new.withColumn("DAYS_NEXT_VISIT", datediff(to_date(df_icu_new.NEXT_VISIT),to_date(df_icu_new.OUTTIME)))

# response column to indicate ICU revisit: can define the time window in days for the next ICU visit to be 
# considered a revisit
# revisit_window = 30
df_icu_new = df_icu_new.withColumn(
    "ICU_REVISIT",
    F.when((F.col('DAYS_NEXT_VISIT') <= 30) & (F.col('DAYS_NEXT_VISIT') >= 0), 1 )
          .otherwise(0))

"""
Merge
"""
# Merge two tables
df_notes_valid = df_notes_valid.drop('SUBJECT_ID')
df_icu_notes = df_icu_new.join(df_notes_valid, on=['HADM_ID'], how='left')

# Drop the rows where CHARTDATE of the note is not between INTIME and OUTTIME, i.e. note wasn't
# generated during ICU stay of a patient
df_icu_notes = df_icu_notes.drop((df_icu_notes.CHARTDATE < df_icu_notes.INTIME) |(df_icu_notes.CHARTDATE>df_icu_notes.OUTTIME))

# Drop any NA rows
df_icu_notes = df_icu_notes.dropna()

# Sort the dataframe by ICUSTAY_ID and CHARTDATE
df_icu_notes.sort(['ICUSTAY_ID', 'CHARTDATE'])

# Get last 10 notes
df_icu_notes.registerTempTable("icu_note")
df_icu_last10 = spark.sql("SELECT *, ROW_NUMBER() OVER (PARTITION BY ICUSTAY_ID ORDER BY CHARTDATE desc) row_num from icu_note")
df_icu_last10 = df_icu_last10.filter(df_icu_new.row_num <= 10)

# Select icuid, label and text
df_icu_last10 = df_icu_last10.select([c for c in df_icu_last10.columns if c in ['ICUSTAY_ID','ICU_REVISIT','TEXT']])
df = df_icu_last10.groupby(['ICUSTAY_ID','ICU_REVISIT']).agg(F.collect_list('TEXT').alias('TEXT'))

"""
Convert to numpy array for modeling
"""
X_n_notes = np.array(df.select('TEXT').collect())
y_n_notes = np.array(df.select('ICU_REVISIT').collect())

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_n_notes, y_n_notes, stratify=y_n_notes, test_size=0.15)


# create a validation set
X_t_n, X_v_n, y_t_n, y_v_n = train_test_split(X_train_n, y_train_n, stratify=y_train_n, test_size=0.1)

"""
Get the GPU support
"""
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# instantiate the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
# model = BertModel.from_pretrained('bert-base-uncased').to(device)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# function for generating feature vector matrix from the ICU visit notes
def features_matrix(X):
    num_stays = X.shape[0]
    seqs = []
    # put the clinicalBERT model in the evaluation mode, meaning FF operation
    model.eval()
    with torch.no_grad():
      for i in range(num_stays):
          ICU_notes = X[i] # list of notes
          num_notes = len(ICU_notes)
          notes_mat = np.zeros((num_notes, 768))
          for j in range(num_notes):
              text = ICU_notes[j]
              
              # tokenize the text with the clinicalBERT tokenizer and add '[CLS]' and '[SEP]' tokens
              tokenized_text = ['[CLS]'] + tokenizer.tokenize(text)[:510] + ['[SEP]']

              tokens_tensor = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_text), device=device).unsqueeze(0)
              encoded_output, _ = model(tokens_tensor)

              # encoded_output[0,0,:] is the feature vector of [CLS] token
              # torch.mean(encoded_output, axis=1)[0] is averaging or pooling the sequence of hidden-states for the whole input sequence
              notes_mat[j,:] = torch.mean(encoded_output, axis=1)[0].cpu().numpy()
              
          seqs.append(sparse.csr_matrix(notes_mat))
        
    return seqs

# generate transformed sequences
train_seqs = features_matrix(X_t_n)
validation_seqs = features_matrix(X_v_n)
test_seqs = features_matrix(X_test_n)

PATH_OUTPUT = "/dbfs/foobar/"

# store sequences
pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(y_t_n.tolist(), open(os.path.join(PATH_OUTPUT, "labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(y_v_n.tolist(), open(os.path.join(PATH_OUTPUT, "labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(y_test_n.tolist(), open(os.path.join(PATH_OUTPUT, "labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
