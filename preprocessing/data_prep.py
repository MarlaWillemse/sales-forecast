from preprocessing.data_utils import *
import os
from root import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import itertools

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

raw = "/data/raw/"
input = 'DS - case study 1 - add material - sales_volumes.csv'
data = pd.read_csv(root + raw + input)

'''Exploratory data analysis'''

# eda(data)

'''Remove unnecessary columns'''

data = data.drop('Unnamed: 0', axis=1)
data = data.drop('Description', axis=1)
data = data.drop('InvoiceID', axis=1)

'''Create box plots to observe outliers'''

boxplot(data, col='UnitPrice', saveas='price_boxplot')
boxplot(data, col='Volume', saveas='volume_boxplot')

'''Remove entries where Volume is beyond [-10000, 10000]
or Value is above 200,000. This is a subjective choice. '''

data = data[(data.Volume > -5000) & (data.Volume < 5000)]
data = data[data.UnitPrice < 200000]

boxplot(data, col='UnitPrice', saveas='price_boxplot_no_outliers')
boxplot(data, col='Volume', saveas='volume_boxplot_no_outliers')

'''Remove time from date and convert to date type'''

data['Date'] = data['Date'].str[:11]
data['Date'] = pd.to_datetime(data['Date'])

'''Join missing dates'''

'''List of dates between start and end date'''
start_dt = date(2019, 1, 1)
end_dt = date(2019, 6, 30)
all_dates = []
for dt in daterange(start_dt, end_dt):
    all_dates.append(dt.strftime("%Y-%m-%d"))

'''List of unique ProductCodes'''
unique_products = data.ProductCode.unique()

'''All permutations of dates and ProductCodes'''
combos = list(itertools.product(all_dates, unique_products))

del all_dates
del unique_products

'''list to df'''
combos = pd.DataFrame(combos, columns=['Date', 'ProductCode'])

'''Same date format to merge on'''
combos.Date = pd.to_datetime(combos.Date)
combos.Date = pd.to_datetime(combos.Date)

'''Merge'''
data = pd.merge(combos, data, how='left',
                left_on=['Date', 'ProductCode'],
                right_on=['Date', 'ProductCode'])
del combos

'''Replace UnitPrice == 0 with nan, to be imputed with mean. 
Assuming that items aren't free'''

data['UnitPrice'] = data['UnitPrice'].replace(0, None)

'''Fill NAN values: impute Volume with zero and UnitPrice with the 
average per ProductCode'''

data.Volume = data.Volume.fillna(value=0)

data['UnitPrice'] = data['UnitPrice'] \
    .fillna(data.groupby(['ProductCode'])['UnitPrice']
            .transform('mean'))

'''Per unique Date and ProductCode, sum Volume 
and average UnitPrice'''

data = data.groupby(['Date', 'ProductCode']) \
    .agg({'Volume': 'sum', 'UnitPrice': 'mean'})

'''Reset Multi-Index to columnar format'''

data = data.reset_index()

'''Downcast numeric types to reduce memory use'''
data = downcast_dtypes(data)

# '''Normalize'''
# data = normalize(data, 'Volume')
# data = normalize(data, 'UnitPrice')

data.to_csv(root + "/data/interim/data_xgboost.csv", header=True,
            index=False)

'''Sanity check: view data for one ProductCode'''

sanity_check1 = data.loc[data['ProductCode'] == '10002']
sanity_check1.to_csv(root + "/data/interim/sanity_check1.csv",
                     header=True, index=False)

del sanity_check1
del data

'''Erase matplotlib parameters'''
plt.clf()
plt.close("all")
