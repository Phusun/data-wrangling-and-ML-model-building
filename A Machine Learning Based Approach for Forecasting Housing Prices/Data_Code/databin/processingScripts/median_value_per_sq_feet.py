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
            if row['state_name'] in states_list[i]:
                exists = 1

    return exists

# Function to replace metro area names to corresponding ones from the standard list of names
def std_name(row):
    for metro in metros_list:
        if metro in row:
            return metro

# read file and drop rows with any missing values
slr = pd.read_excel('Metro_MedianValuePerSqft_AllHomes.xlsx', skiprows=[1])
# slr.dropna(inplace=True)

# split and store area and state
slr[['metro_name', 'state_name']] = slr['RegionName'].apply(lambda row: pd.Series(extract_values(row)))

# identify 50 metro areas in our list
slr['in50'] = slr.apply(metro_matching, axis=1)

# Only retain rows for 50 metro areas in our list
slr = slr[slr['in50']==1]

# create a column with names from our list for consistency
slr['Metro name'] = slr['metro_name'].apply(std_name)

# drop the unnecessary columns and bring the last column first
slr.drop(columns=['SizeRank', 'RegionID', 'RegionName','metro_name', 'state_name', 'in50'], inplace=True)

# Make Metro name column first
cols = list(slr.columns)
cols = [cols[-1]] + cols[:-1]
slr = slr[cols]
slr.set_index('Metro name', inplace=True)
# Set Metro name as index


# Transpose the dataframe and reset index
slr = slr.T
slr.reset_index(inplace=True)
# Add a Year column
slr.rename(columns={'index': 'Period'}, inplace=True)
slr['Year'] = slr['Period'].apply(lambda x: x.split('-')[0])

# Compute yearly average
slr = slr.groupby('Year').mean()

# Transpose and reset index
slr = slr.T
slr.reset_index(inplace=True)

# sort the rows according to metro names alphabetically
slr.sort_values(by=['Metro name'], inplace=True)
slr.reset_index(drop=True, inplace=True)
del slr.columns.name

# write to excel file
w = ExcelWriter('Median Value Per Square Feet.xlsx')
slr.to_excel(w)
w.save()