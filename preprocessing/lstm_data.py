'''Pivot data so that each item is represented for each date.
Sum Volume if product is represented more than once and fill missing
values with zero.'''

data.sort_values(by=['Date', 'ProductCode'])

data_pivot = data.pivot_table(index=['ProductCode'],
                              values=['Volume'],
                              columns=['Date'],
                              aggfunc=np.sum,
                              fill_value=0)