import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.optimizers import SGD




df = pd.read_csv("data/prices-split-adjusted.csv")



#There is 51 number of different stocks company, those below are some of these company.
#['AAPL', 'CLX', 'ETR', 'MCK', 'WMT', 'HCN', 'CTSH', 'NVDA', 'AIV', 'EFX']
# You can specify below any name you want to predict

stock = 'AAPL'
price = 'close' # you can also specify any price you want between those :
# ['open', 'high', 'low', 'close']

df1 = df[df['symbol'] == stock].copy()
df1=df1[[price]]
#df=df.sort_values('date')
#df['date'] = df['date'].apply(pd.to_datetime)  


# normalizing the values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(df1)

x_train = []
y_train = []
timestamp = 60
length = len(df1)
for i in range(timestamp, length):
    x_train.append(training_set_scaled[i-timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# The GRU architecture
regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))


regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')

# fitting the model

regressorGRU.fit(x_train, y_train, epochs=5, batch_size=150)


test_set = df.loc[df['symbol'] == 'CLX']   # change CBS to whatever company from the list
test_set = test_set.loc[:, test_set.columns == price]

y_test = test_set.iloc[timestamp:, 0:].values

closing_price = test_set.iloc[:, 0:].values
closing_price_scaled = sc.transform(closing_price)

x_test = [] 
length = len(test_set)

for i in range(timestamp, length):
    x_test.append(closing_price_scaled[i-timestamp:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


y_pred = regressorGRU.predict(x_test)
predicted_price = sc.inverse_transform(y_pred)

# plotting the results
plt.plot(y_test, color = 'blue', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'red', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
