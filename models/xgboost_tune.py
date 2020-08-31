import os
from root import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import xgboost
from xgboost import XGBRegressor
from preprocessing.data_utils import *

#pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

train_x = pd.read_csv(root+"/data/interim/train_x.csv")
train_y = pd.read_csv(root+"/data/interim/train_y.csv")
test_x = pd.read_csv(root+"/data/interim/test_x.csv")
test_y = pd.read_csv(root+"/data/interim/test_y.csv")

# TODO: Hyperparameter tuning

'''Define the model'''

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

'''Train the model'''

model = model.fit(
    train_x,
    train_y,
    eval_metric="rmse",
    eval_set=[(train_x, train_y), (test_x, test_y)],
    verbose=True,
    early_stopping_rounds=10)

'''Save model to file'''

pickle.dump(model, open(root+"/models_trained/xgboost_1.pickle.dat", "wb"))
#model = pickle.load(open(root+"/models_trained/xgboost_1.pickle.dat", "rb"))

'''Plot feature importance'''

plt.bar(train_x.columns, model.feature_importances_)
plt.suptitle('XGBoost Feature Importance')
plt.xticks(rotation='82.5')

plt.savefig(root+'/reports/figures/xgboost_feat_imp.png', dpi=400)

'''Make predictions'''

preds = model.predict(test_x)

'''Plot predictions against test_y'''

'''Shape data: reconstruct date'''
test_preds = test_x[['Month', 'Day']].copy()
test_preds = reconstruct_date(test_preds)
test_preds = test_preds.sort_values(by=['Date'])
'''Merge predictions and true volume'''
test_preds['Preds'] = pd.DataFrame(preds)
test_preds['Volume'] = test_y.copy()

# '''Inverse transform normalization'''
#
# test_preds = unnormalize(test_preds, 'Preds')
# test_preds = unnormalize(test_preds, 'Volume')

'''Sum volume (over all products) per date'''
test_preds = test_preds.groupby(['Date']).sum()
test_preds = test_preds.reset_index()

test_preds.to_csv(root+"/data/interim/test_preds_plot.csv",
                  header=True, index=False)

'''RMSE of predictions summed per day'''

test_xgboost_rmse = np.sqrt(mean_squared_error(test_preds['Volume'],
                                               test_preds['Preds']))
print(f'Test XGBoost RMSE daily forecast: {test_xgboost_rmse}')

plt.clf()
plt.close("all")