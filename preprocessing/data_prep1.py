from preprocessing.data_utils import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

pd.set_option('display.max_columns', 100)

raw_data_directory = "../data/raw/"
data = pd.read_csv(
    '{0}DS - case study 1 - add material - sales_volumes.csv'.format(
        raw_data_directory))

# TODO: Separate functions into data_utils and calls into run_data_prep

'''Exploratory data analysis'''

eda(data)

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

data_pivot.to_pickle("../data/interim/data_lstm.pkl")

'''Unstack the pivot'''

data_pivot = data_pivot.unstack().reset_index(name='Volume')
data_pivot = data_pivot[['Date', 'ProductCode', 'Volume']]

'''Merge to add the UnitPrice'''

data = pd.merge(data_pivot,
                data[['ProductCode', 'Date', 'UnitPrice']],
                on=['ProductCode', 'Date'],
                how='left')

del data_pivot

# TODO: Consider no invoice for products not sold on given day.
#  For given product on a given day, volume is summed
#  Feature for products sold together?

'''Add UnitPrice values for days when product wasn't sold by filling
with the mean UnitPrice for the ProductCode. '''

data['UnitPrice'] = data.groupby(['ProductCode'], sort=False) \
    ['UnitPrice'].apply(lambda x: x.fillna(x.mean()))

data = downcast_dtypes(data)

data.to_pickle("../data/interim/data_xgboost.pkl")

del data

# TODO: Normalize?



