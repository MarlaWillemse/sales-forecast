from sklearn.metrics import mean_squared_error
sales_m_test = sales_m[sales_m['date_block_num']==33]

preds = sales_m.copy()
preds['date_block_num']=preds['date_block_num']+1
preds = preds[preds['date_block_num']==33]
preds = preds.rename(columns={'item_cnt_day':'preds_item_cnt_day'})
preds = pd.merge(sales_m_test,preds,on = ['shop_id','item_id'],how='left')[['shop_id','item_id','preds_item_cnt_day','item_cnt_day']].fillna(0)

# We want our predictions clipped at (0,20). Competition Specific
preds['item_cnt_day'] = preds['item_cnt_day'].clip(0,20)
preds['preds_item_cnt_day'] = preds['preds_item_cnt_day'].clip(0,20)
baseline_rmse = np.sqrt(mean_squared_error(preds['item_cnt_day'],preds['preds_item_cnt_day']))

print(baseline_rmse)