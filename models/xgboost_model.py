import pandas as pd
import xgboost
from xgboost import XGBRegressor

train_x = pd.read_pickle("../data/interim/train_x.pkl")
train_y = pd.read_pickle("../data/interim/train_y.pkl")
test_x = pd.read_pickle("../data/interim/test_x.pkl")
test_y = pd.read_pickle("../data/interim/test_y.pkl")

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42)

model.fit(
    train_x,
    train_y,
    eval_metric="rmse",
    eval_set=[(train_x, train_y), (test_x, test_y)],
    verbose=True,
    early_stopping_rounds=10)

