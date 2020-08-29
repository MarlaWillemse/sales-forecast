from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from preprocessing.data_utils import *

train_x = pd.read_pickle("../data/interim/train_x.pkl")
train_y = pd.read_pickle("../data/interim/train_y.pkl").fillna(value=0)
test_x = pd.read_pickle("../data/interim/test_x.pkl")
test_y = pd.read_pickle("../data/interim/test_y.pkl").fillna(value=0)

train_preds = train_x['Vol_t-1'].fillna(value=0)
test_preds = test_x['Vol_t-1'].fillna(value=0)

train_baseline_rmse = np.sqrt(mean_squared_error(train_y, train_preds))
test_baseline_rmse = np.sqrt(mean_squared_error(test_y, test_preds))

print(f'Train baseline RMSE: {train_baseline_rmse}')
print(f'Test baseline RMSE: {test_baseline_rmse}')