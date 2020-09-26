import os
from root import *
import xgboost
from xgboost import XGBRegressor
import pickle
import pandas as pd
import datetime
from preprocessing.data_utils import *
from datetime import datetime, timedelta

pd.set_option('display.max_columns', 100)

train_all_x = pd.read_csv(root+"/data/interim/train_all_x.csv")
train_all_y = pd.read_csv(root+"/data/interim/train_all_y.csv")
future_input = pd.read_csv(root+"/data/interim/future_input.csv")
test_preds_plot = pd.read_csv(root+"/data/interim/test_preds_plot.csv")

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
    train_all_x,
    train_all_y,
    eval_metric="rmse",
    verbose=True,
    #early_stopping_rounds=10
)

'''save model to file'''

pickle.dump(model, open(root+"/models_trained/xgboost_2.pickle.dat",
                        "wb"))

model = pickle.load(open(root+"/models_trained/xgboost_2.pickle.dat",
                         "rb"))

'''Initialize pred, preds, and loopdate'''

input_t1 = future_input['Volume']
future_input = future_input.drop('Volume', axis=1)
preds = pd.DataFrame(columns=['Date', 'Future_Volume'])
loopdate = datetime.strptime('2019-06-30', '%Y-%m-%d')
future_input.Date = pd.to_datetime(future_input.Date)

'''Predict 1 day at a time, append prediction, and use to predict
future values'''

while future_input['Date'].iloc[0] < datetime.strptime('2019-07-31',
                                                       '%Y-%m-%d'):

    '''Before each prediction round: shift date and lag variables
    forward by 1 day'''

    '''Shift date forward by 1 day'''
    future_input['Date'] = future_input['Date'] + timedelta(days=1)

    '''Shift lag vars forward by 1 day'''
    for lag in list(range(2, 31)):
        future_input[f'Vol_t-{lag}'] = future_input[f'Vol_t-{lag-1}']

    '''Append previous prediction as t-1'''
    future_input['Vol_t-1'] = input_t1

    '''Create Month and Day columns'''
    future_input = expand_date(future_input)

    '''Limit number of lag variables '''
    limited_vars = ['Vol_t-1', 'Vol_t-2', 'Vol_t-3',
                'Vol_t-4', 'Vol_t-5', 'Vol_t-6', 'Vol_t-7',
                'Vol_t-14', 'Vol_t-21', 'Vol_t-28', 'UnitPrice',
                'ProductCode_ordinal', 'Month', 'Day', 'Weekday']
    future_input_limited = future_input[limited_vars]

    '''Make predictions'''
    pred = model.predict(future_input_limited)

    '''Use predictions as input for next time step'''
    input_t1 = pred

    '''Append pred to preds'''
    pred_append = pd.DataFrame(pred, columns=['Future_Volume'])

    '''Reconstruct Date from Month and Day'''
    future_input = reconstruct_date(future_input)
    pred_append['Date'] = future_input['Date']
    loopdate = future_input['Date'].iloc[0]
    preds = pd.concat([preds, pred_append])

# '''Inverse transform normalization'''
# preds = unnormalize(preds, 'Future_Volume')

'''Sum volume (over all products) per date'''

preds = preds.groupby(['Date']).sum()
preds = preds.reset_index()
preds.to_csv(root+"/data/processed/predictions.csv", header=True,
             index=False)

'''Concatenate future predictions to past values and test predictions'''

preds_plot = preds.copy()
preds_plot['Preds'] = np.nan
preds_plot['Volume'] = np.nan
preds_plot = preds_plot[['Date', 'Volume', 'Preds', 'Future_Volume']]
test_preds_plot['Future_Volume'] = np.nan
test_preds_plot = test_preds_plot[['Date', 'Volume', 'Preds',
                                   'Future_Volume']]
test_preds_plot = pd.concat([test_preds_plot, preds])

'''Date format for matplotlib'''

test_preds_plot.Date = pd.to_datetime(test_preds_plot.Date)
test_preds_plot = test_preds_plot.sort_values(by=['Date'])
test_preds_plot['Date'] = test_preds_plot['Date']\
    .apply(lambda x: x.strftime('%Y-%m-%d'))

'''Plot'''

plt.figure(figsize=(17, 8))
plt.plot(test_preds_plot['Date'], test_preds_plot['Volume'],
         label="Past Volume")
plt.plot(test_preds_plot['Date'], test_preds_plot['Preds'],
         label="Test Predictions")
plt.plot(test_preds_plot['Date'], test_preds_plot['Future_Volume'],
         label="Future Predictions")
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Past sales, test predictions, and future predictions')
plt.legend()
plt.xticks(rotation='50')
plt.savefig(root+"/reports/figures/test_preds_vs_true.png")
plt.clf()
plt.close("all")
plt.show()