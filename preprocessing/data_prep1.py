from preprocessing.data_utils import *
import os
from root import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

pd.set_option('display.max_columns', 100)

raw = "/data/raw/"
data = pd.read_csv(root+raw+'DS - case study 1 - add material - sales_volumes.csv')

'''Exploratory data analysis'''

#eda(data)

'''Remove unnecessary columns'''

data = data.drop('Unnamed: 0', axis=1)
data = data.drop('Description', axis=1)

'''Create box plots to observe outliers'''

boxplot(data, col='UnitPrice', saveas='price_boxplot')
boxplot(data, col='Volume', saveas='volume_boxplot')

'''Remove entries where Volume is beyond [-10,000; 10;000}
or Value is above 200,000. This is a subjective choice. '''

data = data[(data.Volume > -10000) & (data.Volume < 10000)]
data = data[data.UnitPrice < 200000]

boxplot(data, col='UnitPrice', saveas='price_boxplot_no_outliers')
boxplot(data, col='Volume', saveas='volume_boxplot_no_outliers')

# TODO remove entry where price differs greatly from product average

'''Remove time from date and convert to date type'''

data['Date'] = data['Date'].str[:11]
# # Could keep time:
# data['Time'] = data['Date'].str[11:16]
data['Date'] = pd.to_datetime(data['Date'])

'''Join missing dates'''

'''List of dates between start and end date'''
start_dt = date(2019, 1, 1)
end_dt = date(2019, 6, 30)
all_dates = []
for dt in daterange(start_dt, end_dt):
    all_dates.append(dt.strftime("%Y-%m-%d"))
'''list to df'''
all_dates = pd.DataFrame(all_dates, columns=['Date'])
'''Same date format to merge on'''
data.Date = pd.to_datetime(data.Date)
all_dates.Date = pd.to_datetime(all_dates.Date)
'''Merge'''
data = pd.merge(all_dates, data, on='Date', how='left')

'''Pivot data so that each item is represented for each date.
Sum Volume if product is represented > once and fill missing values
with zero.'''

data.sort_values(by=['Date', 'ProductCode'])

data_pivot = data.pivot_table(index=['ProductCode'],
                              values=['Volume'],
                              columns=['Date'],
                              aggfunc=np.sum,
                              fill_value=0)

data_pivot = downcast_dtypes(data_pivot)

data_pivot.to_pickle(root+"/data/interim/data_lstm.pkl")

'''Unstack the pivot'''

data_pivot = data_pivot.unstack().reset_index(name='Volume')
data_pivot = data_pivot[['Date', 'ProductCode', 'Volume']]

'''Merge to add the UnitPrice'''

data = pd.merge(data_pivot,
                data[['ProductCode', 'Date', 'UnitPrice']],
                on=['ProductCode', 'Date'],
                how='left')

del data_pivot

# TODO: Feature encodings for ProductCode and Invoice

'''Add UnitPrice values for days when product wasn't sold by filling
with the mean UnitPrice for the ProductCode. '''

data['UnitPrice'] = data.groupby(['ProductCode'], sort=False) \
    ['UnitPrice'].apply(lambda x: x.fillna(x.mean()))

data = downcast_dtypes(data)

data.to_pickle(root+"/data/interim/data_xgboost.pkl")

# TODO: Normalize

plt.clf()
plt.close("all")

