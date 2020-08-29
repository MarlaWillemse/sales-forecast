import xgboost
from xgboost import XGBRegressor
import pickle
import pandas as pd

train_all_x = pd.read_csv("../data/interim/train_all_x.csv", header=None)
train_all_y = pd.read_csv("../data/interim/train_all_y.csv", header=None)
train_future = pd.read_csv("../data/interim/future_input.csv", header=None)
train_future_header = pd.read_csv("../data/interim/future_input_header.csv", header=None)

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
#     train_all_x,
#     train_all_y,
#     eval_metric="rmse",
#     verbose=True,
#     #early_stopping_rounds=10
# )
#
# '''save model to file'''
#
# pickle.dump(model, open("../models_trained/xgboost_2.pickle.dat", "wb"))

model = pickle.load(open("../models_trained/xgboost_2.pickle.dat", "rb"))

#TODO Create input for predictions:
# shift dates forward, e.g. Volume = t-1

print(train_future_header.head)