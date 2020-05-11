import pandas as pd
import re
from pandas import ExcelWriter

metros = pd.read_excel('Metro names.xlsx', header=0)
metros_list = list(metros.iloc[:,0])
states_list = list(metros.iloc[:,1])

# Function to extract the metro area name and total units through regex
def extract_values(row):
    s = re.findall('(.+)\,\s+(\w{2}[\-\w{2}]*)', row)
    metro, state = s[0]
    return (metro, state)

# Function to match metro and state names
def metro_matching(row):
    exists = 0
    for i, metro in enumerate(metros_list):
        if metro in row['metro_name']:
            if row['state_name'] == states_list[i]:
                exists = 1

    return exists

# Function to replace metro area names to corresponding ones from the standard list of names
def std_name(row):
    for metro in metros_list:
        if metro in row:
            return metro

# read data from two sheets
HPI_AL = pd.read_excel('House Price Index/Freddie Mac House Price Index.xls', sheet_name='MSA Indices SA A-L', header=5, skipfooter=17)

HPI_MZ = pd.read_excel('House Price Index/Freddie Mac House Price Index.xls', sheet_name='MSA Indices SA M-Z', header=5, skipfooter=17)

# main function
def dfCreator(df):
    # drop any columns with all NaN
    df.dropna(axis=1, how='all', inplace=True)

    # add a Year column
    df['Year'] = df['Month'].apply(lambda row: row.split('M')[0])

    # move the Year column to the front
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]

    # drop the Month column, not needed now
    df.drop(columns=['Month'], inplace=True)

    # compute the mean of all columns by year
    df_mean = df.groupby('Year').mean()

    # transpose so that Year: columns and metros: rows
    df_T = df_mean.T
    # reset index, delete the column name and rename the place column
    df_T.reset_index(inplace=True)
    del df_T.columns.name
    df_T.rename(columns={'index': 'Place'}, inplace=True)

    # split and store area and state
    df1 = df_T.copy()
    df1[['metro_name', 'state_name']] = df1['Place'].apply(lambda row: pd.Series(extract_values(row)))

    # identify 50 metro areas in our list
    df1['in50'] = df1.apply(metro_matching, axis=1)

    # Only retain rows for 50 metro areas in our list
    df1 = df1[df1['in50']==1]

    # create a column with names from our list for consistency
    df2 = df1.copy()
    df2['Metro name'] = df1['metro_name'].apply(std_name)

    # drop the unnecessary columns and bring the last column first
    df2.drop(columns=['Place','metro_name', 'state_name', 'in50'], inplace=True)

    cols = list(df2.columns)
    cols = [cols[-1]] + cols[:-1]
    df2 = df2[cols]

    return df2

# create individual dataframes and concatenate them along the index
df_AL = dfCreator(HPI_AL)
df_MZ = dfCreator(HPI_MZ)

df_joined = pd.concat([df_AL, df_MZ], ignore_index=True)

# write it in an Excel file
w = ExcelWriter('House Price Index.xlsx')
df_joined.to_excel(w)
w.save()