import pandas as pd
import xgboost
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pickle
from xgboost import Booster

pd.set_option('display.max_rows', 100)

train_x = pd.read_csv("../data/interim/train_x.csv", header=None)
train_y = pd.read_csv("../data/interim/train_y.csv", header=None)
test_x = pd.read_csv("../data/interim/test_x.csv", header=None)
test_y = pd.read_csv("../data/interim/test_y.csv", header=None)

# # TODO: Hyperparameter tuning
#
# '''Define the model'''
#
# model = XGBRegressor(
#     max_depth=8,
#     n_estimators=1000,
#     min_child_weight=300,
#     colsample_bytree=0.8,
#     subsample=0.8,
#     eta=0.3,
#     seed=42)
#
# '''Train the model'''
#
# model = model.fit(
#     train_x,
#     train_y,
#     eval_metric="rmse",
#     eval_set=[(train_x, train_y), (test_x, test_y)],
#     verbose=True,
#     early_stopping_rounds=10)
#
# '''save model to file'''
#
# pickle.dump(model, open("../models_trained/xgboost_1.pickle.dat", "wb"))
#
# '''Plot feature importance'''
#
# plt.bar(train_x.columns, model.feature_importances_)
# plt.suptitle('XGBoost Feature Importance')
# plt.xticks(rotation='82.5')
#
# plt.savefig('../reports/figures/xgboost_feat_imp.png', dpi=400)
#
# '''Make predictions'''
#
# model = pickle.load(open("../models_trained/xgboost_1.pickle.dat", "rb"))
# preds = model.predict(test_x)
# df = pd.DataFrame(preds)
# print(df)
#
# '''Plot predictions against test_y'''
#
# # test set with header
# test_x = pd.read_csv("../data/interim/test_x_header.csv")
#
# # Shape data: reconstruct date
# test_preds = test_x[['Month', 'Day']].copy()
# test_preds['Year'] = '2019'
# test_preds['Month'] = test_preds['Month'].apply(str)
# test_preds['Day'] = test_preds['Day'].apply(str)
# test_preds['Date'] = test_preds['Year'].str\
#     .cat(test_preds['Month'], sep="-")
# test_preds['Date'] = test_preds['Date'].str\
#     .cat(test_preds['Day'], sep="-")
# test_preds = test_preds.drop('Year', axis=1)
# test_preds = test_preds.drop('Month', axis=1)
# test_preds = test_preds.drop('Day', axis=1)
# test_preds['Date'] = pd.to_datetime(test_preds.Date)
# test_preds = test_preds.sort_values(by=['Date'])
# # Merge predictions and true volume
# test_preds['Preds'] = pd.DataFrame(preds)
# test_preds['Volume'] = test_y.copy()
# # Sum volume (over all products) per date
# test_preds = test_preds.groupby(['Date']).sum()
# test_preds = test_preds.reset_index()
# print(test_preds)
#
# # Plot
# plt.plot(test_preds['Date'], test_preds['Volume'], label="True Volume")
# plt.plot(test_preds['Date'], test_preds['Preds'], label="Predictions")
# plt.xlabel('Date')
# plt.ylabel('Volume')
# plt.title('True a predicted volume: test set')
# plt.legend()
# plt.xticks(rotation='82.5')
# plt.savefig(f"../reports/figures/test_preds_vs_true.png")
# plt.show()

'''Train the model using all data'''

model = model.fit(
    train_x,
    train_y,
    eval_metric="rmse",
    eval_set=[(train_x, train_y), (test_x, test_y)],
    verbose=True,
    early_stopping_rounds=10)

'''save model to file'''

pickle.dump(model, open("../models_trained/xgboost_1.pickle.dat", "wb"))