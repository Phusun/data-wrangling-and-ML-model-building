import pandas as pd
import numpy as np
from pandas import ExcelWriter

df = pd.read_excel('Clean Data/Real Per Capita Personal Income.xlsx', usecols=lambda x: 'Unnamed' not in x)

df1 = df.drop(['Metro name'], axis=1)

def extrp2018(row):
    x = np.arange(2008, 2018, dtype='float32')
    y = row.to_numpy(dtype='float32')
    poly = np.polyfit(x, y, deg=1)
    return round(np.polyval(poly, 2018))

df['2018'] = df1.apply(lambda row: extrp2018(row), axis=1)

# write to excel file
w = ExcelWriter('Clean Data/Real Per Capita Personal Income 2018.xlsx')
df.to_excel(w)
w.save()