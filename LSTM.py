import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def drift(y, n):
    y_res = list(np.nan for i in range(0, len(y)))
    for i in range(2, n):
        y_res[i] = y[i-1] + ((y[i-1]-y[0]))/(i-1)
    for i in range(n, len(y)):
        y_res[i] = y[n-1] + (i+1-n)*(y[n-1]-y[0])/(n-1)
    return y_res

def non_seasonal_differencing(column, n):
    difference = []
    for i in range(0, n):
        difference.append(np.nan)
    for i in range(n, len(column)):
        difference.append(column[i] - column[i-1])
    return difference

def seasonal_differencing(series, seasons=1):
    diff = []
    for i in range(seasons):
        diff.append(np.nan)
    for i in range(seasons, len(series)):
        diff.append(series[i] - series[i - seasons])
    return diff

# Define a function to create the LSTM dataset
def lstm_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def cal_errors(y, y_pred, n, skip_first_n_obs=0):
    e = []
    e_sq = []
    for i in range(0, len(y)):
        if y_pred[i] != np.nan:
            e.append(y[i] - y_pred[i])
            e_sq.append((y[i] - y_pred[i]) ** 2)
        else:
            e.append(np.nan)
            e_sq.append(np.nan)
    mse_tr = np.nanmean(e_sq[skip_first_n_obs:n])
    var_pred = np.nanvar(e[skip_first_n_obs:n])
    mse_ts = np.nanmean(e_sq[n:])
    var_fcst = np.nanvar(e[n:])
    res_mean = np.nanmean(e[skip_first_n_obs:n])
    return e, e_sq, mse_tr, var_pred, mse_ts, var_fcst, res_mean


np.random.seed(6313)

url = 'https://raw.githubusercontent.com/sshikha78/Solar-Radiation-Prediction/main/SolarPrediction.csv'
df = pd.read_csv(url)

df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Time'])
df.sort_values(by='Datetime', inplace=True)
df = df.drop(['UNIXTime'], axis=1)
print(df.head().to_string())
print(df.info())
df['TimeSunRise'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunRise']).astype(np.int64)
df['TimeSunSet'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunSet']).astype(np.int64)
df_hourly = df.resample('H', on='Datetime').mean()


#==============================================
#   Check NA value    #####
#===============================================

print(f'NA value: \n{df_hourly.isna().sum()}')
df_hourly.reset_index(inplace=True)
print(df_hourly.head().to_string())
print(df_hourly[df_hourly.isna().any(axis=1)])
col_list = df_hourly.columns[df_hourly.isna().any()].tolist()
index_list = df_hourly[df_hourly.isna().any(axis=1)].index.tolist()

# Filling NA with forecasted value of drift method  #
for col in col_list:
    for index in index_list:
        y_pred = drift(df_hourly[col], index)
        df_hourly[col][index] = y_pred[index]


#==============================================
# Checking if NA remain AFTER DRIFT      ####
#===============================================

print(f'NA value: \n{df_hourly.isna().sum()}')
df_hourly.set_index('Datetime', inplace=True)


#==============================================
# Plotting dependent variable vs time   ####
#===============================================
s=24
df_hourly['TimeSunRise'] = pd.to_datetime(df_hourly['TimeSunRise'], format='%Y-%m-%d %H:%M:%S')
df_hourly['TimeSunSet'] = pd.to_datetime(df_hourly['TimeSunSet'], format='%Y-%m-%d %H:%M:%S')
date = pd.date_range(start = '2016-09-01', periods = len(df_hourly), freq='H')
df_hourly.index = date
df = df_hourly
df['seasonal_d_o_1'] = seasonal_differencing(df['Radiation'], seasons=s)
df['diff_order_1'] = non_seasonal_differencing(df['seasonal_d_o_1'], s)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df[['diff_order_1']][s+1:])

# Split the dataset into df_train and df_test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
df_train, df_test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

x_train, y_train = lstm_dataset(df_train, look_back=3)
x_test, y_test = lstm_dataset(df_test, look_back=3)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = Sequential()
model.add(LSTM(4, input_shape=(1, 3)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2, callbacks=[early_stopping], validation_split=0.1)

y_prediction = model.predict(x_train)
y_forecast = model.predict(x_test)
y_prediction = scaler.inverse_transform(y_prediction)
y_train = scaler.inverse_transform([y_train])
y_forecast = scaler.inverse_transform(y_forecast)
y_test = scaler.inverse_transform([y_test])

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(df['diff_order_1'][s+1:len(df_train)].index.values, df['diff_order_1'][s+1:len(df_train)].values, label='df_train Data')
plt.plot(df['diff_order_1'][s+1+len(df_train):].index.values, df['diff_order_1'][s+1+len(df_train):].values, label='df_test Data')
plt.plot(df['diff_order_1'][s+1+len(df_train)+2+2:].index.values, y_forecast, label='Predictions')

plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('LSTM')
plt.legend()
plt.tight_layout()
plt.show()

y_train1 = pd.Series(y_train[0])
y_test1 = pd.Series(y_test[0])
trainPredict2 = pd.Series(y_prediction[:, 0])
testPredict2 = pd.Series(y_forecast[:, 0])
print(pd.concat([trainPredict2, testPredict2]))

df_err = pd.DataFrame(list(zip(pd.concat([y_train1, y_test1], axis=0), pd.concat([trainPredict2, testPredict2], axis=0))), columns=['y', 'y_pred'])
print(df_err)

e, e_sq, mse_tr, var_pred, mse_ts, var_fcst, res_mean = cal_errors(df_err['y'].to_list(), df_err['y_pred'].to_list(), len(y_train1), 0)

q_value = sm.stats.acorr_ljungbox(e[:len(y_train1)], lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
print('Error values using LSTM method')
print('MSE Prediction data: ', round(mse_tr, 2))
print('MSE Forecasted data: ', round(mse_ts, 2))
print('Variance Prediction data: ', round(var_pred, 2))
print('Variance Forecasted data: ', round(var_fcst, 2))
print('res_mean: ', round(res_mean, 2))
print('Q-value: ', round(q_value, 2))
var_f_vs_r = round(var_fcst / var_pred, 2)
print(f'var(forecast errors)/var(Residual errors): {var_f_vs_r:.2f}')

lstm_data = [['LSTM', mse_tr, mse_ts, var_pred, var_fcst, res_mean, q_value, var_f_vs_r]]
df_model_comp = pd.DataFrame(lstm_data, columns=['method_name', 'mse_tr', 'mse_ts', 'var_pred', 'var_fcst', 'res_mean', 'Q-value', 'var_fcst vs var_pred'])
print(df_model_comp)

