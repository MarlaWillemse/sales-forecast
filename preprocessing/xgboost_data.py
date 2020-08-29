from preprocessing.data_utils import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

# pd.set_option('display.max_columns', 100)
data = pd.read_pickle("../data/interim/data_xgboost.pkl")

# TODO: Optimize n
n = 8
lags = list(range(1, (n+1)))
for lag in lags:
    data[f'Vol_t-{lag}'] = data.sort_values('Date').groupby\
    (['ProductCode', 'UnitPrice'])['Volume'].shift(lag)

'''Remove first n dates for which not all lag variables are available'''

data = data[data['Date'] > (data.Date.min() + timedelta(days=n))]

'''Replace ProductCode with ordinal encodings'''

# TODO: Replace ordinal encoding with learned feature embeddings with
#  invoice and ProductCode

data['ProductCode_ordinal'] = LabelEncoder().\
    fit_transform(data['ProductCode'])

data = data.drop('ProductCode', axis=1)

# TODO: Perhaps create lag variables

'''Sort on date and ProductCode'''
data = data.sort_values(['Date', 'ProductCode_ordinal'])

'''Train-test split'''

nr_dates = data['Date'].nunique()
train_proportion = int(nr_dates*0.7)
train_cutoff_date = data.Date.min() + \
                    timedelta(days=(n+train_proportion))

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

train_x['Month'] = pd.DatetimeIndex(train_x['Date']).month*1
train_x['Day'] = pd.DatetimeIndex(train_x['Date']).day*1
train_x = train_x.drop('Date', axis=1)
test_x['Month'] = pd.DatetimeIndex(test_x['Date']).month*1
test_x['Day'] = pd.DatetimeIndex(test_x['Date']).day*1
test_x = test_x.drop('Date', axis=1)
train_all_x['Month'] = pd.DatetimeIndex(train_all_x['Date']).month*1
train_all_x['Day'] = pd.DatetimeIndex(train_all_x['Date']).day*1
train_all_x = train_all_x.drop('Date', axis=1)
future['Month'] = pd.DatetimeIndex(future['Date']).month*1
future['Day'] = pd.DatetimeIndex(future['Date']).day*1
future = future.drop('Date', axis=1)

'''Create prediction input'''

train_x.to_csv("../data/interim/train_x_header.csv", header=True, index=False)
train_y.to_csv("../data/interim/train_y_header.csv", header=True, index=False)
test_x.to_csv("../data/interim/test_x_header.csv", header=True, index=False)
test_y.to_csv("../data/interim/test_y_header.csv", header=True, index=False)

# Don't feed header to model!
train_x.to_csv("../data/interim/train_x.csv", header=False, index=False)
train_y.to_csv("../data/interim/train_y.csv", header=False, index=False)
test_x.to_csv("../data/interim/test_x.csv", header=False, index=False)
test_y.to_csv("../data/interim/test_y.csv", header=False, index=False)

train_all_x.to_csv("../data/interim/train_all_x.csv", header=False, index=False)
train_all_y.to_csv("../data/interim/train_all_y.csv", header=False, index=False)

future.to_csv("../data/interim/future_input.csv", header=False, index=False)
future.to_csv("../data/interim/future_input_header.csv", header=True, index=False)