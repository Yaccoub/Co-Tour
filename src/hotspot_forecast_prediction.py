from datetime import datetime
import math

import keras
import matplotlib.pyplot as plt
import mlflow.tensorflow
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import folium
from folium import plugins
from geopy.geocoders import Nominatim


def create_sequences(dataset, timesteps=1, dropNa=True):
    # drop row's which include Nan elements (data preprocessing)
    df = pd.DataFrame(dataset)
    if dropNa:
        df.dropna(inplace=True)
    dataset = df.values
    # create x and y out of dataset
    dataX, dataY = [], []
    for i in range(len(dataset)):
        endIdx = i + timesteps
        # stop if reached the end of dataset
        if endIdx + 1 > len(dataset):
            break
        dataX.append(dataset[i:endIdx, :])
        dataY.append(dataset[endIdx, :])
    return np.array(dataX), np.array(dataY)


def test_train(datasetsize, testsize, shuffle=True):
    if shuffle:
        ntest = int(np.ceil(testsize * datasetsize))
        idx = np.arange(0, datasetsize)
        np.random.shuffle(idx)
        test_index = idx[:ntest]
        return test_index
    else:
        ntest = 1  # int(np.ceil(testsize * datasetsize))
        idx = np.arange(0, datasetsize)
        test_index = idx[datasetsize - ntest:]
        return test_index


np.random.seed(7)
dataset = pd.read_csv('../data/Forecast Data/dataset.csv')
dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
dataset = dataset.set_index('DATE')

timesteps = 20
dropNan = False
shuffle = False
for i in range(2, -1, -1):
    dataset_d = dataset.drop(dataset.index[len(dataset) - i:len(dataset)])
    for place in dataset_d.columns[2:4]:
        reconstructed_model = keras.models.load_model("../ML_models/{}.h5".format(place))
        X_Data, y_Data = create_sequences(dataset_d[place], timesteps, dropNan)
        y_Data = pd.DataFrame(y_Data)

        y_Data = y_Data.rename(columns={0: place})
        y_Data_c = y_Data.set_index(dataset_d.index[timesteps:])

        test_index = test_train(len(X_Data), 0.33, shuffle)

        # rename the columns of y_Data
        X_test = X_Data[test_index]
        y_test = y_Data.loc[test_index]
        predictions = reconstructed_model.predict(X_test)

        valid = pd.DataFrame(y_test)
        valid['Predictions'] = predictions
        dataset[place][dataset.index[len(dataset) - i - 1]] = predictions

dataset = dataset.reset_index()
dataset = dataset.rename(columns={"index": "DATE"})
dataset.to_csv('../data/Forecast Data/dataset_predicted.csv', index=False)