import os
from root import *
from preprocessing.data_utils import *
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 100)

data = pd.read_pickle(root+"/data/interim/data_xgboost.pkl")

'''Create lag variables: volume at previous time points'''

n = 28
lags = list(range(1, (n+1)))
for lag in lags:
    data[f'Vol_t-{lag}'] = data.sort_values('Date').groupby\
    (['ProductCode', 'UnitPrice'])['Volume'].shift(lag)

'''Remove first n dates for which not all lag variables are available'''

data = data[data['Date'] > (data.Date.min() + timedelta(days=max(lags)))]

'''Replace ProductCode with ordinal encodings'''

# TODO: Replace ordinal encoding with learned feature embeddings with
#  invoice and ProductCode

data['ProductCode_ordinal'] = LabelEncoder().\
    fit_transform(data['ProductCode'])

data = data.drop('ProductCode', axis=1)

'''Sort on date and ProductCode'''
data = data.sort_values(['Date', 'ProductCode_ordinal'])

'''Fill missing values with zero'''

data = data.fillna(value=0)

'''Create Weekday variable'''
data['Weekday'] = data['Date'].dt.dayofweek

'''Train-test split'''

nr_dates = data['Date'].nunique()
train_proportion = int(nr_dates*0.7)
train_cutoff_date = data.Date.min() + \
                    timedelta(days=(max(lags)+train_proportion))

train = data[data['Date'] >= train_cutoff_date]
test = data[data['Date'] < train_cutoff_date]

'''Input for future predictions'''

future = data[data['Date'] == (data.Date.max())]

'''Separate the target values from the predictors'''

train_y = train['Volume']
train_x = train.drop('Volume', axis=1)
del train
test_y = test['Volume']
test_x = test.drop('Volume', axis=1)
del test
train_all_y = data['Volume']
train_all_x = data.drop('Volume', axis=1)
del data

'''Replace Date with a month and day ordinal variables'''

train_x = month_and_day_from_date(train_x)
test_x = month_and_day_from_date(test_x)
train_all_x = month_and_day_from_date(train_all_x)

'''To avoid overfitting, limit the number of lag variables: use only 
1, 2, 3, 7, 14, 21, and 28 days ago'''

limited_vars = ['UnitPrice', 'Vol_t-1', 'Vol_t-2', 'Vol_t-3',
                   'Vol_t-7', 'Vol_t-14', 'Vol_t-21', 'Vol_t-28',
                   'ProductCode_ordinal', 'Month', 'Day', 'Weekday']

train_x = train_x[limited_vars]
test_x = test_x[limited_vars]
train_all_x = train_all_x[limited_vars]

'''Create prediction input'''

train_x.to_csv(root+"/data/interim/train_x.csv", header=True,
               index=False)
train_y.to_csv(root+"/data/interim/train_y.csv", header=True,
               index=False)
test_x.to_csv(root+"/data/interim/test_x.csv", header=True,
              index=False)
test_y.to_csv(root+"/data/interim/test_y.csv", header=True,
              index=False)

train_all_x.to_csv(root+"/data/interim/train_all_x.csv", header=True,
                   index=False)
train_all_y.to_csv(root+"/data/interim/train_all_y.csv", header=True,
                   index=False)

future.to_csv(root+"/data/interim/future_input.csv", header=True,
              index=False)
