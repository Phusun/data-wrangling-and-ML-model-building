import csv
import os
import pandas as pd
import numpy as np

data_dir = 'Clean Macro data/'
yfile = 'Sale to List Ratio.csv'
metro_name_key = 'Metro name'
allFiles = os.listdir(data_dir)
csvs = []
features = []
for myFile in allFiles:
    if myFile[-4:] == '.csv':
        csvs.append(myFile)
        features.append(myFile[:-4])

dfs = []
for dataFile in csvs:
    # if dataFile == yfile:
    #     continue
    df = pd.read_csv(data_dir + dataFile)
    df = df[[metro_name_key, '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']]
    df = df.sort_values(by=[metro_name_key])
    df = df[['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']]
    df = df.transpose()
    # Learned about unstack here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html
    df = df.unstack()
    # Learned about droplevel here: https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index
    df = df.droplevel(level=0)
    dfs.append(df)

# Learned about concat here: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
feature_table = pd.concat(dfs, axis=1)
# Learned about rename here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
for i in range(len(features)):
    feature_table = feature_table.rename(columns={i: features[i]})

needed_cols = [metro_name_key, '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
df = pd.read_csv(data_dir + yfile)
df = df[needed_cols]
df = df.sort_values(by=[metro_name_key])
metroNames = df[metro_name_key]
needed_cols = needed_cols[2:]
df = df[needed_cols]
nans = np.zeros((df.shape[0],2))
nans[:, :] = float('nan')
new_df = pd.DataFrame(data=nans, columns=['2020', '2021'])
merged = np.hstack((df.values, new_df.values))
new_headers = np.append(df.columns, new_df.columns)
df = pd.DataFrame(data=merged, index=df.index, columns=new_headers)

one_year = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
two_year = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
three_year = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

one_df = df[one_year]
one_df = one_df.transpose()
# Learned about unstack here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html
one_df = one_df.unstack()
# Learned about droplevel here: https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index
one_df = one_df.droplevel(level=0)

two_df = df[two_year]
two_df = two_df.transpose()
# Learned about unstack here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html
two_df = two_df.unstack()
# Learned about droplevel here: https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index
two_df = two_df.droplevel(level=0)

three_df = df[three_year]
three_df = three_df.transpose()
# Learned about unstack here: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.unstack.html
three_df = three_df.unstack()
# Learned about droplevel here: https://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index
three_df = three_df.droplevel(level=0)

metroNames = pd.concat([metroNames] * int(feature_table.shape[0] / metroNames.shape[0]), axis=1)
metroNames = metroNames.transpose()
metroNames = metroNames.unstack()
metroNames = metroNames.droplevel(level=0)
# print(metroNames)
merged = np.hstack((feature_table.values, one_df.values[:, np.newaxis], two_df.values[:, np.newaxis], three_df.values[:, np.newaxis], metroNames.values[:, np.newaxis]))
new_headers = np.append(feature_table.columns, ['change next year', 'change 2 years', 'change 3 years', metro_name_key])
feature_table = pd.DataFrame(data=merged, index=feature_table.index, columns=new_headers)

feature_table.sort_index(inplace=True)

feature_table.to_csv(path_or_buf='ml_dataset.csv')
