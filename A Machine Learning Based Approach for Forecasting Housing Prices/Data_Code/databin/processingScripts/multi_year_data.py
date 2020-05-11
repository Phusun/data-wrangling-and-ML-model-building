import pandas as pd
import numpy as np

# read feature files
df1 = pd.read_excel('Clean Data/Building Units.xlsx', usecols=lambda x: 'Unnamed' not in x)
df2 = pd.read_excel('Clean Data/Employees Total Nonfarm.xlsx', usecols=lambda x: 'Unnamed' not in x)
df3 = pd.read_excel('Clean Data/House Price Index.xlsx', usecols=lambda x: 'Unnamed' not in x)
df4 = pd.read_excel('Clean Data/Real Per Capita Personal Income.xlsx', usecols=lambda x: 'Unnamed' not in x)
df5 = pd.read_excel('Clean Data/Resident Population.xlsx', usecols=lambda x: 'Unnamed' not in x)
df6 = pd.read_excel('Clean Data/Unemployment Rate.xlsx', usecols=lambda x: 'Unnamed' not in x)

df_list = [df1, df2, df3, df4, df5, df6]
features_list = ['Buildings', 'Employees', 'HPI', 'Income', 'Population', 'Unemployment']

# get all the data for a year
year = '2010'
df_joined = None

for i in range(len(df_list)):
    if df_joined is None:
        df_joined = df_list[i][[year]]
        df_joined.rename(columns={year: features_list[i]}, inplace=True)
    else:
        df_joined[features_list[i]] = df_list[i][[year]]

# Flattened numpy array of shape (300,)
# df_array = df_joined.to_numpy().ravel()


