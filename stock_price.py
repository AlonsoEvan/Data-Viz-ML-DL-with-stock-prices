import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import functions

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)



df = pd.read_csv("data/prices-split-adjusted.csv")

#There is 51 number of different stocks company, those below are some of these company.
#['AAPL', 'CLX', 'ETR', 'MCK', 'WMT', 'HCN', 'CTSH', 'NVDA', 'AIV', 'EFX']
# You can specify below any name you want to predict

stock = 'AAPL'
price = 'close' # you can also specify any price you want between those :
# ['open', 'high', 'low', 'close']
#WARNING, you need to specify the same price in the file functions.py

df = df[df['symbol'] == stock]
df=df[['date', price]]
df=df.sort_values('date')
df['date'] = df['date'].apply(pd.to_datetime)


#moving average

df['EMA_9'] = df[price].ewm(9).mean().shift()
df['SMA_5'] = df[price].rolling(5).mean().shift()
df['SMA_10'] = df[price].rolling(10).mean().shift()
df['SMA_15'] = df[price].rolling(15).mean().shift()
df['SMA_30'] = df[price].rolling(30).mean().shift()


#relative strenght index

df['RSI'] = functions.relative_strength_idx(df).fillna(0)


#shift label column

df[price] = df[price].shift(-1)


#drop invalid samples, we calculated moving average so the first rows are NaN

df = df.iloc[33:] # Because of moving averages and MACD line
df = df[:-1]      # Because of shifting close price

df.index = range(len(df))


#build train, test and valid data

test_size  = 0.15
valid_size = 0.15

test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df  = df.loc[:valid_split_idx]
valid_df  = df.loc[valid_split_idx+1:test_split_idx]
test_df   = df.loc[test_split_idx+1:]


train_df = train_df[[price, 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI']]
valid_df = valid_df[[price, 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI']]
test_df = test_df[[price, 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI']]


y_train = train_df[price]
X_train = train_df.drop([price], axis = 1)

y_valid = valid_df[price]
X_valid = valid_df.drop([price], axis = 1)

y_test  = test_df[price]
X_test  = test_df.drop([price], 1)

eval_set = [(X_train, y_train), (X_valid, y_valid)]


#our model :

model = xgb.XGBRegressor(gamma = 0.01, learning_rate = 0.05, max_depth =  8, n_estimators = 400, random_state = 42, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=2)


#if you want to plot feature importance, uncomment the two line of code below

#plot_importance(model)
#plt.show()


#predictions
y_pred = model.predict(X_test)

print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')


predicted_prices = df.loc[test_split_idx+1:]
predicted_prices[price] = y_pred



#print prediction in comparison to actual values to see if our model perform well
fig = plt.figure(figsize = (8,6))

ax1 = fig.add_subplot(211)

ax1.plot(df['date'], df[price], label = 'Truth')

ax1.plot(predicted_prices['date'], predicted_prices[price], label = 'Prediction')
ax1.set_xlabel('time')
ax1.set_ylabel(price)
ax1.legend(loc = 'best')


ax2 = fig.add_subplot(212)

ax2.plot(predicted_prices['date'], y_pred, label = 'Truth')

ax2.plot(predicted_prices['date'], y_test, label = 'Prediction')
ax2.set_xlabel('time')
ax2.set_ylabel(price)
ax2.legend(loc = 'best')

plt.show()

