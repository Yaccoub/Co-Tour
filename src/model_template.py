import pandas as pd
import numpy as np
from datetime import datetime


def create_sequences(dataset, timesteps=1, dropNa=True):
    # drop row's which include Nan elements (data preprocessing)
    df = pd.DataFrame(dataset)
    if dropNa == True:
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
    if shuffle == True:
        ntest = int(np.ceil(testsize * datasetsize))
        idx = np.arange(0, datasetsize)
        np.random.shuffle(idx)
        train_index = idx[ntest:]
        test_index = idx[:ntest]
        return train_index, test_index
    else:
        ntest = int(np.ceil(testsize * datasetsize))
        idx = np.arange(0, datasetsize)
        train_index = idx[ntest:]
        test_index = idx[:ntest]
        return train_index, test_index


def main():
    # importing the data
    df = pd.read_csv('../data/dataset/test.csv')

    # change some variable formats
    df['Datum'] = [datetime.strptime(date, '%Y-%m-%d') for date in df['Datum']]
    df = df.set_index('Datum')

    # convert the time-series to an array of sequences
    timesteps = 2
    dropNan = False
    X_Data, y_Data = create_sequences(df, timesteps, dropNan)

    # rename columns of y dataset
    y_Data = pd.DataFrame(y_Data)
    for old_name, new_name in zip(y_Data.columns, df.columns):
        y_Data = y_Data.rename(columns={old_name: new_name})
    y_Data

    # test train split
    testsize = 0.33
    shuffle = True
    train_index, test_index = test_train(len(X_Data), testsize, shuffle)

    # rename the columns of y_Data

    X_train = X_Data[train_index]
    X_test = X_Data[test_index]
    y_train = y_Data.loc[train_index]
    y_test = y_Data.loc[test_index]


if __name__ == '__main__':
    main()
