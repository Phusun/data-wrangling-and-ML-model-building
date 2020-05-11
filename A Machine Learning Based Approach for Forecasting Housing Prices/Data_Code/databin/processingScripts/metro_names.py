import pandas as pd

metros = pd.read_excel('Metro names.xlsx', header=0)
metros_list = list(metros.iloc[:,0])
states_list = list(metros.iloc[:,1])

personal_income = pd.read_excel('Per Capita Personal Income 2015-2017.xlsx', header=None, skiprows=8, usecols=[0,1,2,3], names=['Metropolitan Areas','2015','2016','2017'])
personal_income.dropna(inplace=True)


# Split the metro areas and their states
personal_income['metro_name'] = personal_income['Metropolitan Areas'].apply(lambda x: x.split(',')[0])
personal_income['state_name'] = personal_income['Metropolitan Areas'].apply(lambda x: x.split(',')[1].lstrip()) # remove the leading whitespace from state abbreviations

# Function to match metro and state names
def metro_matching(row):
    exists = 0

    for i, metro in enumerate(metros_list):
        if metro in row['metro_name']:
            if row['state_name'] == states_list[i]:
                exists = 1

    return exists

personal_income['in50'] = personal_income.apply(metro_matching, axis=1)

# Only retain rows for 50 metro areas in our list
personal_income = personal_income[personal_income['in50']==1]

# Function to replace metro area names to corresponding ones from the standard list of names
def std_name(row):
    for metro in metros_list:
        if metro in row:
            return metro

personal_income_metros = personal_income.copy()
personal_income_metros['Metro name'] = personal_income['metro_name'].apply(std_name)

# drop columns, reset index and rearrange the columns
personal_income_metros.drop(columns=['Metropolitan Areas', 'metro_name', 'state_name', 'in50'], inplace=True)
personal_income_metros.reset_index(drop=True, inplace=True)
personal_income_metros = personal_income_metros[['Metro name', '2015', '2016', '2017']]

print(personal_income_metros)