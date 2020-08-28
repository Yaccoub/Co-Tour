from datetime import datetime
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv1D, MaxPooling1D, LSTM
from keras.callbacks import EarlyStopping
from keras.losses import mean_squared_error
import mlflow.keras
from mlflow.entities import Param,Metric,RunTag
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
        ntest = int(np.ceil(testsize * datasetsize))
        idx = np.arange(0, datasetsize)
        test_index = idx[datasetsize - ntest:]
        train_index = idx[:datasetsize - ntest]
        return train_index, test_index


def LSTM_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity']
                 )

def build_model_cnn(n_steps,n_feats,n_fore=1):
    model = Sequential()
    model.add(Conv1D(filters=50, kernel_size=3, activation='relu',input_shape=(n_steps,n_feats)))
    #model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dropout(0.20))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(n_fore, activation='linear'))
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity']
                 )
    return model

# Create Experiment in Mlflow for tracking
try:
    experiment_id = mlflow.create_experiment(name='1D-CNN')
    print("Created mlflow experiment")
except:
    experiment_id = mlflow.get_experiment_by_name(name='1D-CNN').experiment_id

# Set random seed for data splitting
np.random.seed(7)
dataset = pd.read_csv('../data/Forecast Data/dataset.csv')
dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
dataset = dataset.set_index('DATE')

# Throw an exception when containing NaN values
if dataset.isnull().sum().sum() != 0:
    raise Exception("The dataset contains NaN values")

# Iterator for testing
#for iterator in [1,2,8,16,32,64,128]:
    #print("Test model with ",iterator)
# Switch of the iterator
if True:
    # Preprocessing setup
    n_steps = 5
    dropNan = False
    shuffle = True

    X_Data, y_Data_comp = create_sequences(dataset, n_steps, dropNan)
    # Drop the featurs that we don't want to predict
    y_Data = y_Data_comp[:,:28]

    # Test train split
    train_index, test_index = test_train(len(X_Data), 0.33, shuffle)

    # rename the columns of y_Data
    X_train = X_Data[train_index]
    X_test = X_Data[test_index]
    y_train = y_Data[train_index]
    y_test = y_Data[test_index]

    # Parameters for model setup
    n_feats = X_train.shape[2]
    n_fore = y_Data.shape[1]

    # Setup model logging
    run_name = "final_test"
    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

    mlflow.keras.autolog()

    # Create and build the model
    model = build_model_cnn(n_steps,n_feats,n_fore)
    model.summary()

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=1,
        epochs=1000,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(patience=10),
        ],
    )

    # Save the model
    model.save('../ML_models/{}.h5'.format("1D-CNN"))

    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mse')
    plt.ylabel('mean suqared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    Figure = "../img/val_mean_squared_error.png"
    fig.savefig(Figure)
    #plt.show()
    mlflow.log_artifact(Figure)

    fig = plt.figure()
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mae')
    plt.ylabel('mean absolute error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    Figure = "../img/val_mean_absolute_error.png"
    fig.savefig(Figure)
    #plt.show()
    mlflow.log_artifact(Figure)

    fig = plt.figure()
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title('model mean absolute percentage error')
    plt.ylabel('mean absolute percentage error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    Figure = "../img/val_mean_absolute_percentage_error.png"
    fig.savefig(Figure)
    #plt.show()
    mlflow.log_artifact(Figure)

    fig = plt.figure()
    plt.plot(history.history['cosine_proximity'])
    plt.plot(history.history['val_cosine_proximity'])
    plt.title('model cosine proximity')
    plt.ylabel('cosine proximity')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    Figure = "../img/val_cosine_proximity.png"
    fig.savefig(Figure)
    #plt.show()
    mlflow.log_artifact(Figure)

    # Create Predictions
    predictions = model.predict(X_test)

    # Visualize the predicted data
    fig = plt.figure()
    plt.scatter(y_test,predictions)
    plt.grid(True)
    plt.title("correlation prediction - test data")
    plt.xlabel("actual test data")
    plt.ylabel("predictions")
    Figure = "../img/correlation.png"
    fig.savefig(Figure)
    #plt.show()
    mlflow.log_artifact(Figure)

    # Calculate Pearson's correlation
    li = []
    for i in np.arange(predictions.shape[1]):
        corr, _ = pearsonr(predictions[i],y_test[i])
        li.append(corr)
    # Calcuate the mean Pearson's correlation
    corr_mean = np.mean(li)
    print("Mean Pearsons correlation: %.3f" % corr_mean)
    mlflow.log_param("pearson_correlation", corr_mean)

    # End logging
    mlflow.end_run()
