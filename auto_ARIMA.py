#  garbage collector
import gc

import pandas as pd
import numpy as np
from pyramid.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

import Load_Data as LD

log_trans = True

air_visit_data = LD.air_visit_data
data = pd.pivot_table(air_visit_data, values = 'visitors', index = 'air_store_id', columns = 'visit_date', fill_value = 0)
# Normalization with log1p
if log_trans:
    data = np.log(1+data)
data.drop(index = ['air_0ead98dd07e7a82a',
                   'air_229d7e508d9f1b5e',
                   'air_2703dcb33192b181',
                   'air_b2d8bc9c88b85f96',
                   'air_cb083b4789a8d3a2',
                   'air_cf22e368c1a71d53',
                   'air_d0a7bd3339c3d12a',
                   'air_d63cfa6d6ab78446'], inplace = True)
# series = pd.DataFrame(data.T.values, index = data.columns)
# print(series)
print(data)

N = data.shape[0]
answer = np.zeros((N,39))
train = True
if train:
    # Apply ARIMA on each restaurant
    for i in range(N):
        train_data = data.iloc[i]
        # fit model
        model = auto_arima(train_data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        # freq : str {'B','D','W','M','A', 'Q'}
        #     'B' - business day, ie., Mon. - Fri.
        #     'D' - daily
        #     'W' - weekly
        #     'M' - monthly
        #     'A' - annual
        #     'Q' - quarterly
        # model = ARIMA(train_data, order=(7,1,0), freq = 'D')
        model.fit(train_data)
        answer[i,:] = model.predict(n_periods=39)
        del model
        gc.collect()
        # model_fit.predict(start = '2017-04-23', end = '2017-05-31')
        if (i+1)%(N/10.) < 1:
            print('{:.0%} have done!'.format((i+1)/N))

    np.save('answer.npy', answer)
    print(answer)

# answer = np.load('answer.npy')
# Denormalize
if log_trans:
    answer = np.exp(answer)-1

# # validation
# def valid():
#     return

# submit.csv
def submit(answer):
    test_range = pd.date_range(start='2017-04-23', end='2017-05-31', freq='D')
    with open('submit.csv', 'w') as f:
        f.write('id,visitors\n')
        for i,r in enumerate(data.index):
            for a, date in zip(answer[i], test_range):
                f.write('{}_{:%Y-%m-%d},{}\n'.format(r, date, a))

submit(answer)
print('Done!')

# plot residual errors
# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot()
# plt.savefig("1.png")
# plt.clf()
# residuals.plot(kind='kde')
# plt.savefig("2.png")
# plt.clf()
# print(residuals.describe())
