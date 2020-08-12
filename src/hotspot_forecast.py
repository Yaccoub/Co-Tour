import glob
from pathlib import Path
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import dateparser
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

Covid_19 = pd.read_csv('../data/covid_19_data/rki/COVID_19_Cases_SK_Muenchen.csv', low_memory=False)
dataset = pd.read_csv('../data/dataset/Dataset.csv', low_memory=False)


#Covid_19["Refdatum"] = pd.to_datetime(Covid_19['Refdatum'], errors='coerce')
Covid_19['Refdatum']= [datetime.strptime(date, '%Y-%m-%d')for date in Covid_19['Refdatum']]

Covid_19 = Covid_19.set_index('Refdatum')
Covid_19 =Covid_19.resample('1M').sum()
Covid_19.index =Covid_19.index + timedelta(days=1)
dataset['DATE']= [datetime.strptime(date, '%Y-%m-%d')for date in dataset['DATE']]
dataset = dataset.set_index('DATE')

dataset = pd.concat([dataset,Covid_19], axis=1)
dataset['AnzahlFall'] = dataset['AnzahlFall'].fillna(0)
#
#Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
#
np.random.seed(7)
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
#
#
data = dataset.filter(['Olympiapark'])
data2 = data.values
data_covid = dataset.filter(['AnzahlFall'])
data_covid = data_covid.values
test_size = 15
train_size = int(len(dataset) ) - test_size
x_train=[]
x_cov=[]
y_train = []
for i in range(20,train_size):
    x_train.append(data2[i-20:i,0])
    x_cov.append(data_covid[i - 20:i, 0])
    y_train.append(data2[i,0])

x_train, y_train, x_cov = np.array(x_train), np.array(y_train), np.array(x_cov)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_cov = np.reshape(x_train, (x_cov.shape[0],x_cov.shape[1],1))
x_train = np.concatenate((x_train, x_cov), axis=2)
#

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100)

test_data = data2[train_size - 20:, :]

#Create the x_test and y_test data sets
x_test = []
y_test =  data2[train_size :  ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(20,len(test_data)):
    x_test.append(test_data[i-20:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Getting the models predicted price values
predictions = model.predict(x_test)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

#Plot/Create the data for the graph
train = data[:train_size]
valid = pd.DataFrame(data[train_size:])
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train)
plt.plot(valid[['Olympiapark', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()