import pandas as pd
from pandas import ExcelWriter

metros = pd.read_excel('Metro names.xlsx', header=0)
metros_list = list(metros.iloc[:,0])
states_list = list(metros.iloc[:,1])

file_names = ['Unemployment Rate.csv', 'Resident Population.csv', 'Real Per Capita Personal Income.csv', 'Employees Total Nonfarm.csv']

df_joined = None

for i in file_names[3:4]:
    for j in metros_list:
        location = 'METRO AREAS/'+j+'/'+i
        df = pd.read_csv(location, sep=',', header=None, names=[j], skiprows=1, usecols=[1])

        if df_joined is None:
            df_joined = df
        else:
            df_joined[j] = df[j]

# transponse to switch metro names and years
df_T = df_joined.T

# dictionary for renaming the years
years = [str(i) for i in range(1990, 2019)]

years_dict = {k:v for k,v in enumerate(years)}

# rename the columns for the years
df_T.rename(columns=years_dict, inplace=True)
df_T.reset_index(inplace=True)
df_T.rename(columns={'index': 'Metro name'}, inplace=True)
# sort metro names alphabetically
df_T.sort_values(by=['Metro name'], inplace=True)
df_T.reset_index(drop=True, inplace=True)


# write to excel file
w = ExcelWriter(i.split('.')[0]+'.xlsx')
df_T.to_excel(w)
w.save()