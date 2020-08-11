import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt


mod = sm.tsa.statespace.SARIMAX(state['Außenanlagen Olympiapark (Veranstaltungen)'][:'01/02/2020'], order=(1,0,1))
res = mod.fit(disp=False)


nforecast = 20
predict = res.get_prediction(end=mod.nobs + nforecast)
idx = np.arange(len(predict.predicted_mean))
predict_ci = predict.conf_int(alpha=0.5)

# Graph
fig, ax = plt.subplots(figsize=(12,6))
ax.xaxis.grid()
ax.plot(state['Außenanlagen Olympiapark (Veranstaltungen)'][:'01/02/2020'], 'k.')

# Plot
ax.plot(idx[:-nforecast], predict.predicted_mean[:-nforecast], 'gray')
ax.plot(idx[-nforecast:], predict.predicted_mean[-nforecast:], 'k--', linestyle='--', linewidth=2)
ax.fill_between(idx, predict_ci[:, 0], predict_ci[:, 1], alpha=0.15)

ax.set(title='Figure 8.9 - Internet series');