from datetime import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

X = dataset.loc[:, dataset.columns != 'Olympiapark']
Y = dataset['Olympiapark']
test_size = 15
train_size = int(len(dataset)) - test_size
train_X, train_y = X[:train_size].dropna(), Y[:train_size].dropna()
test_X, test_y = X[train_size:], Y[train_size:]

seas_d = sm.tsa.seasonal_decompose(X['Olympiastadion'].dropna(), model='add', freq=12);
fig = seas_d.plot()
fig.set_figheight(4)
plt.show()


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


y_test = Y[:train_size].dropna()
test_adf(y_test, "Olympiapark")
test_adf(y_test.diff(), 'Olympiapark')

fig, ax = plt.subplots(2, 1, figsize=(10, 5))
fig = sm.tsa.graphics.plot_acf(y_test, lags=50, ax=ax[0])
fig = sm.tsa.graphics.plot_pacf(y_test, lags=50, ax=ax[1])
plt.show()

# step_wise=auto_arima(train_y,
#  exogenous= train_X,
#  start_p=1, start_q=1,
#  max_p=7, max_q=7,
#  d=1, max_d=7,
#  trace=True,
#  error_action='ignore',
#  suppress_warnings=True,
#  stepwise=True)
# Statespace


sarimax = sm.tsa.statespace.SARIMAX(train_y,seasonal_order=(3,0,3,12),exog= train_X,
                                enforce_stationarity=False, enforce_invertibility=False).fit()

sarimax.summary()
nforecast = 10
predict = sarimax.get_prediction(end=sarimax.nobs + nforecast, exog=test_X.dropna())
idx = np.arange(len(predict.predicted_mean))
predict_ci = predict.conf_int(alpha=0.5)
train_y = pd.DataFrame(train_y)
fig, ax = plt.subplots(figsize=(12,6))
ax.xaxis.grid()
ax.plot(train_y, 'k.')


# Plot
ax.plot(predict_ci.index[:-nforecast], predict.predicted_mean[:-nforecast], 'gray')
ax.plot(predict_ci.index[-nforecast:], predict.predicted_mean[-nforecast:], 'b-', linestyle='-', linewidth=2)
ax.fill_between(predict_ci.index[50:], predict_ci[predict_ci.index[50:]][predict_ci.columns[0]], predict_ci[predict_ci.index[50:]][predict_ci.columns[1]], alpha=0.15)

#
# model = SARIMAX(train_y,
#                 exog=train_X,
#                 order=(1, 1, 1),
#                 enforce_invertibility=False, enforce_stationarity=False)
# results= model.fit()
# predictions= results.predict(start =train_size, end=train_size+test_size-1,exog=test_X)
# forecast_1= results.forecast(steps=test_size-1, exog=test_X)
# act= pd.DataFrame(scaler_output.iloc[train_size:, 0])
# predictions=pd.DataFrame(predictions)
# predictions.reset_index(drop=True, inplace=True)
# predictions.index=test_X.index
# predictions['Actual'] = act['Stock Price next day']
# predictions.rename(columns={0:'Pred'}, inplace=True)
# predictions['Actual'].plot(figsize=(20,8), legend=True, color='blue')
# predictions['Pred'].plot(legend=True, color='red', figsize=(20,8))