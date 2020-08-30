from datetime import datetime
import keras
import numpy as np
import pandas as pd
from pickle import load

def create_sequences(dataset, in_steps=1, out_steps=1, dropNa=True):
    """Converts time series into a data set for supervised machine learning models"""
    # drop row's which include Nan elements (data preprocessing)
    df = pd.DataFrame(dataset)
    if dropNa:
        df.dropna(inplace=True)
    dataset = df.values
    # create x and y out of dataset
    dataX, dataY = [], []
    for i in range(len(dataset)):
        endIdx = i + in_steps + out_steps
        # stop if reached the end of dataset
        if endIdx > len(dataset):
            break
        dataX.append(dataset[i:endIdx-out_steps, :])
        dataY.append(dataset[endIdx-out_steps:endIdx, :])
    return np.array(dataX), np.array(dataY)

# Reading the dataset
dataset = pd.read_csv('../data/Forecast Data/dataset.csv')
dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
dataset = dataset.set_index('DATE')

# Throw an exception when containing NaN values
if dataset.isnull().sum().sum() != 0:
    raise Exception("The dataset contains NaN values")

# Preprocessing setup
# Attention: This setup must be the same as in hotspot_forecast_train.py except output_len = 0. Otherwise the models wont match.
n_steps = 16
output_len = 0
dropNan = False

# Create dataset for prediction
X_data_all, y_data = create_sequences(dataset, n_steps, output_len, dropNan)
X_data = X_data_all[len(X_data_all)-1]

# Reshape the input data to a 3 dim array
X_data = np.reshape(X_data, (1,X_data.shape[0],X_data.shape[1]))

# Load the xscaler
xscalers = load(open('../data_scaler/xscalers.pkl', 'rb'))

X_data_scaled = X_data
for i in range(X_data.shape[2]):
    X_data_scaled[:, :, i] = xscalers[i].transform(X_data[:, :, i])

# Get the places that we wanna predict
places = dataset.columns[:28]

ret = []

# Iterator over different places
for idx in np.arange(len(places)):
    place = places[idx]
    print("Start prediction for:",place)

    # Load the traind model
    model = keras.models.load_model("../ML_models/{}.h5".format(place))

    # Load the xscaler
    yscaler = load(open('../data_scaler/yscaler/{}.pkl'.format(place), 'rb'))

    # Predict
    y_pred = yscaler.inverse_transform(model.predict(X_data_scaled))

    # Append prediction to the list of different places
    ret.append(pd.DataFrame(np.reshape(y_pred, y_pred.shape[1]), columns=[place]))

# Generate new indices
indices = []
for i in np.arange(4):
    indices.append(dataset.index[-1] + pd.DateOffset(months=i + 1))

# Append timestamps to
ret.append(pd.DataFrame(indices, columns=['DATE']))
predictions = pd.concat(ret, axis=1, sort=True)

# Set date as index
predictions = predictions.set_index('DATE')

# Deal with negative predichtios
predictions[predictions < 0] = 0

#Save file
predictions.to_csv('../data/Forecast Data/dataset_predicted.csv', index=True)
