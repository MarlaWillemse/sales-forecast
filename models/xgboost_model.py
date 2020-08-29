import pandas as pd
import xgboost
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

train_x = pd.read_pickle("../data/interim/train_x.pkl")
train_y = pd.read_pickle("../data/interim/train_y.pkl")
test_x = pd.read_pickle("../data/interim/test_x.pkl")
test_y = pd.read_pickle("../data/interim/test_y.pkl")

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

model.fit(
    train_x,
    train_y,
    eval_metric="rmse",
    eval_set=[(train_x, train_y), (test_x, test_y)],
    verbose=True,
    early_stopping_rounds=10)

'''Plot feature importance'''

plt.bar(train_x.columns, model.feature_importances_)
plt.suptitle('XGBoost Feature Importance')
plt.xticks(rotation='82.5')

plt.savefig('../reports/figures/xgboost_feat_imp.png', dpi=400)
plt.show()

'''Make predictions'''

