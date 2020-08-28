from datetime import datetime

import mlflow.tensorflow
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


def create_sequences(dataset, timesteps=1, dropNa=True):
    """Converts time series into a data set for supervised machine learning models"""
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
    """Returns two dataset to train and test machine learning models"""
    if shuffle:
        ntest = int(np.ceil(testsize * datasetsize))
        idx = np.arange(0, datasetsize)
        np.random.shuffle(idx)
        train_index = idx[ntest:]
        test_index = idx[:ntest]
        return train_index, test_index
    else:
        #TODO: Check datasplitting use int(np.ceil(testsize * datasetsize))
        ntest = 1  # int(np.ceil(testsize * datasetsize))
        idx = np.arange(0, datasetsize)
        test_index = idx[datasetsize - ntest:]
        train_index = idx[:datasetsize - ntest]
        return train_index, test_index


def LSTM_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],2))) # For multivariate lstm
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Create Experiment in Mlflow for tracking
try:
    experiment_id = mlflow.create_experiment(name='Lstm')
except:
    experiment_id = mlflow.get_experiment_by_name(name='Lstm').experiment_id

np.random.seed(7)
dataset = pd.read_csv('../data/Forecast Data/dataset.csv')
dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
dataset = dataset.set_index('DATE')

dataset_d = dataset.drop(dataset.index[len(dataset) - 2:len(dataset)])

timesteps = 20
dropNan = False
shuffle = False
for place in dataset_d.columns[2:4]:
    X_Data, y_Data = create_sequences(dataset_d[place], timesteps, dropNan)
    y_Data = pd.DataFrame(y_Data)

    y_Data = y_Data.rename(columns={0: place})
    y_Data_c = y_Data.set_index(dataset_d.index[timesteps:])

    train_index, test_index = test_train(len(X_Data), 0.33, shuffle)

    # rename the columns of y_Data
    X_train = X_Data[train_index]
    X_test = X_Data[test_index]
    y_train = y_Data.loc[train_index]
    y_test = y_Data.loc[test_index]

    # Logging
    mlflow.start_run(experiment_id=experiment_id, run_name=place)
    # For multivariate lstm
    # mlflow.start_run(run_name='Multivariate LSTM')

    mlflow.tensorflow.autolog()
    # create and fit the LSTM network

    model = LSTM_model()
    model.fit(X_train, y_train, batch_size=1, epochs=2)
    model.save('../ML_models/{}.h5'.format(place))
    mlflow.end_run()