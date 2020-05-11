import pandas as pd
import re
# import openpyxl
from pandas import ExcelWriter

metros = pd.read_excel('Metro names.xlsx', header=0)
metros_list = list(metros.iloc[:,0])
states_list = list(metros.iloc[:,1])

years = [str(i) for i in range(2004,2019)]

# dataframe to store all the years
df_joined = None

# file for writing the data into
w = ExcelWriter('Building Units.xlsx')

# Function to extract the metro area name and total units through regex
def extract_values(row):
    s = re.findall('(.+)\,.?\s+(\w{2}[\-\w{2}]*)\s+(\d+).*', row[0])
    metro, state, units = s[0]
    return (metro, state, int(units))

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

# cycle through all the files
for i in years:
    filename = 'Building Permits/Housing Units/tb3u' + i + '.txt'

    # read files
    housing_unit = pd.read_csv(filename, sep='\t', header=None, names=['data'], skiprows=10, skipfooter=3, engine='python')

    # split and store area, state and units
    df = housing_unit.copy()
    df[['metro_name', 'state_name', 'total_units']] = df.apply(lambda row: pd.Series(extract_values(row)), axis=1)

    # drop the original 'data' column
    df.drop(columns=['data'], inplace=True)

    # identify 50 metro areas in our list
    df['in50'] = df.apply(metro_matching, axis=1)

    # Only retain rows for 50 metro areas in our list
    df = df[df['in50']==1]

    # create a column with names from our list for consistency
    df1 = df.copy()
    df1['Metro name'] = df['metro_name'].apply(std_name)

    # drop the unnecessary columns, reset the index and rename the units column for the year it represents
    df1.drop(columns=['metro_name', 'state_name', 'in50'], inplace=True)
    df1.rename(columns={'total_units': i}, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1 = df1[['Metro name', i]]

    # join all the years together
    if df_joined is None:
        df_joined = df1
    else:
        df_joined[i] = df1[i]

# write to excel file
df_joined.to_excel(w)
w.save()