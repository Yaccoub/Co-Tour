from datetime import datetime
from datetime import timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mlflow.tensorflow
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller



np.random.seed(7)
Covid_19 = pd.read_csv('../data/covid_19_data/rki/COVID_19_Cases_SK_Muenchen.csv', low_memory=False)
Covid_19['Refdatum'] = [datetime.strptime(date, '%Y-%m-%d') for date in Covid_19['Refdatum']]
Covid_19 = Covid_19.set_index('Refdatum')
Covid_19 = Covid_19.resample('1M').sum()
Covid_19.index = Covid_19.index + timedelta(days=1)

dataset = pd.read_csv('../data/Forecast Data/dataset.csv', low_memory=False)
dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
dataset = dataset.set_index('DATE')

dataset = pd.concat([dataset, Covid_19], axis=1)

dataset['AnzahlFall'] = dataset['AnzahlFall'].fillna(0)

data = dataset.filter(['Olympiapark'])
data2 = data.values

# # For multivariate lstm
# data_covid = dataset.filter(['AnzahlFall'])
# data_covid = data_covid.values

test_size = 15
train_size = int(len(dataset)) - test_size

x_train = []
# x_cov=[] # For multivariate lstm
y_train = []

for i in range(20, train_size):
    x_train.append(data2[i - 20:i, 0])
    # x_cov.append(data_covid[i - 20:i, 0]) # For multivariate lstm
    y_train.append(data2[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# For multivariate lstm
# x_cov = np.array(x_cov) # For multivariate lstm
# x_cov = np.reshape(x_train, (x_cov.shape[0],x_cov.shape[1],1))
# x_train = np.concatenate((x_train, x_cov), axis=2)

# Logging
mlflow.tensorflow.autolog()
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],2))) # For multivariate lstm
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = data2[train_size - 20:, :]
x_test = []
y_test = data2[train_size:]


# For multivariate lstm
# test_cov = data_covid[train_size - 20:, :]
# x_test_cov = []

for i in range(20, len(test_data)):
    x_test.append(test_data[i - 20:i, 0])
    # x_test_cov.append(test_cov[i-20:i,0]) # For multivariate lstm

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# # For multivariate lstm
# x_test_cov = np.array(x_test_cov)
# x_test_cov = np.reshape(x_test, (x_test_cov.shape[0],x_test_cov.shape[1],1))
# x_test = np.concatenate((x_test, x_test_cov), axis=2)

# Getting the models predicted price values
predictions = model.predict(x_test)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
#rmse = np.sqrt(mean_squared_error(predictions, y_test))
# Plot/Create the data for the graph
train = data[:train_size]
valid = pd.DataFrame(data[train_size:])
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train)
plt.plot(valid[['Olympiapark', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# # plot history
# pyplot.plot(model.history['loss'], label='train')
# pyplot.plot(model.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
