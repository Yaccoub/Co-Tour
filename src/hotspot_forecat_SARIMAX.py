from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import mlflow.tensorflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def test_adf(series, title=''):
    dfout = {}
    dftest = sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')
    for key, val in dftest[4].items():
        dfout[f'critical value ({key})'] = val
    if dftest[1] <= 0.05:
        print("Strong evidence against Null Hypothesis")
        print("Reject Null Hypothesis - Data is Stationary")
        print("Data is Stationary", title)
    else:
        print("Strong evidence for  Null Hypothesis")
        print("Accept Null Hypothesis - Data is not Stationary")
        print("Data is NOT Stationary for", title)


np.random.seed(7)

# Create Experiment in Mlflow for tracking
try:
    experiment_id = mlflow.create_experiment(name='SARIMAX')
except:
    experiment_id = mlflow.get_experiment_by_name(name='SARIMAX').experiment_id

dataset = pd.read_csv('../data/Forecast Data/dataset.csv', low_memory=False)
dataset['DATE'] = [datetime.strptime(date, '%Y-%m-%d') for date in dataset['DATE']]
dataset = dataset.set_index('DATE')

X = dataset.loc[:, dataset.columns != 'Olympiapark']
Y = dataset['Olympiapark']

test_size = 15
train_size = int(len(dataset)) - test_size
train_X, train_y = X[:train_size].dropna(), Y[:train_size].dropna()
test_X, test_y = X[train_size:], Y[train_size:]
# # Logging
mlflow.start_run(experiment_id=experiment_id, run_name='SARIMAX')
seas_d = sm.tsa.seasonal_decompose(Y.dropna(), model='add');
fig = seas_d.plot()
plt.show()
fig1 = 'Seasonal Decomposition.png'
fig.savefig(fig1)
mlflow.log_artifact(fig1)  # logging to mlflow

y_test = Y[:train_size].dropna()
test_adf(y_test, "Olympiapark")
test_adf(y_test.diff(), 'Olympiapark')
fig, ax = plt.subplots(2, 1, figsize=(10, 5))
fig = sm.tsa.graphics.plot_acf(y_test, lags=50, ax=ax[0])
fig = sm.tsa.graphics.plot_pacf(y_test, lags=50, ax=ax[1])
plt.show()
fig2 = 'ACF-PACF.png'
fig.savefig(fig2)
mlflow.log_artifact(fig2)  # logging to mlflow

ps = 1
qs = 1
trend = 'ct'  # or 'ct'
stationarity = True
invertibility = True
param = [ps, qs, trend, stationarity, invertibility]
# Logging
mlflow.log_param('param-ps', param[0])
mlflow.log_param('param-qs', param[1])
mlflow.log_param('param-trend', param[2])
mlflow.log_param('param-stationality', param[3])
mlflow.log_param('param-invertibility', param[4])
model = SARIMAX(train_y, seasonal_order=(param[0], 0, param[1], 12), exog=train_X, trend=param[2],
                enforce_stationarity=param[3], enforce_invertibility=param[4]).fit()

model.summary()
nforecast = 10
predict = model.get_prediction(end=model.nobs + nforecast, exog=test_X.dropna())
idx = np.arange(len(predict.predicted_mean))
predict_ci = predict.conf_int(alpha=0.5)
train_y = pd.DataFrame(train_y)
fig, ax = plt.subplots(figsize=(12, 6))
ax.xaxis.grid()
ax.plot(train_y, 'k.')

# Plot
ax.plot(predict_ci.index[:-nforecast], predict.predicted_mean[:-nforecast], 'gray')
ax.plot(predict_ci.index[-nforecast:], predict.predicted_mean[-nforecast:], 'b-', linestyle='-', linewidth=2)
ax.fill_between(predict_ci.index[50:], predict_ci[predict_ci.columns[0]][predict_ci.index[50:]],
                predict_ci[predict_ci.columns[1]][predict_ci.index[50:]], alpha=0.15)
Fig1 = 'Forecast.png'
fig.savefig(Fig1)
mlflow.log_artifact(Fig1)  # logging to mlflow
