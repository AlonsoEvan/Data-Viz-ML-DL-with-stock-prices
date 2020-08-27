import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing

import os
import matplotlib.pyplot as plt

import random 
import seaborn as sns
from fbprophet import Prophet

import functions

import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("data/prices-split-adjusted.csv")

'''
df_stock = df[df['symbol'] == 'EQIX']
print(df_stock.head())
df_stock.drop('symbol', axis = 1, inplace = True)
df_stock.drop('volume', axis = 1, inplace = True)

df_stock_norm = functions.normalize_data(df_stock)

seq_len = 20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = functions.load_data(df_stock_norm, seq_len)
'''
'''
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)
'''

#There is 51 number of different stocks company, those below are some of these company.
#['AAPL', 'CLX', 'ETR', 'MCK', 'WMT', 'HCN', 'CTSH', 'NVDA', 'AIV', 'EFX']
# You can specify below any name you want to predict

stock = 'AAPL'
price = 'close' # you can also specify any price you want between those :
# ['open', 'high', 'low', 'close']

df_prophet = df[df['symbol'] == stock]
df_prophet=df_prophet[['date', price]]
df_prophet=df_prophet.sort_values('date')
df_prophet=df_prophet.rename(columns={'date':'ds',price:'y'})



m=Prophet()
m.fit(df_prophet)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)

figure=m.plot(forecast,xlabel='Date',ylabel='{}_Price'.format(price))
plt.show()