
import seaborn as sns
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL

# Set display options for pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def drift(y, n):
    y_pred = list(np.nan for i in range(0, len(y)))
    for i in range(2, n):
        y_pred[i] = y[i-1] + ((y[i-1]-y[0]))/(i-1)
    for i in range(n, len(y)):
        y_pred[i] = y[n-1] + (i+1-n)*(y[n-1]-y[0])/(n-1)
    return y_pred

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

df_hourly['TimeSunRise'] = pd.to_datetime(df_hourly['TimeSunRise'], format='%Y-%m-%d %H:%M:%S')
df_hourly['TimeSunSet'] = pd.to_datetime(df_hourly['TimeSunSet'], format='%Y-%m-%d %H:%M:%S')
date = pd.date_range(start = '2016-09-01', periods = len(df_hourly), freq='H')
df_hourly.index = date
df = df_hourly

def differencing(series, order=1):
    diff = []
    for i in range(order):
        diff.append(np.nan)
    for i in range(order, len(series)):
        diff.append(series[i] - series[i - 1])
    return diff

def seasonal_differencing(series, seasons=1):
    diff = []
    for i in range(seasons):
        diff.append(np.nan)
    for i in range(seasons, len(series)):
        diff.append(series[i] - series[i - seasons])
    return diff

s=24
df['seasonal_d_o_1'] = seasonal_differencing(df['Radiation'], seasons=s)
df['diff_order_1'] = differencing(df['seasonal_d_o_1'], s)

# Import necessary modules for LSTM model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Define a function to create the LSTM dataset
def create_lstm_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Set the target variable
target = 'diff_order_1'

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df[[target]][s+1:])

# Split the dataset into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Reshape the dataset into LSTM format
look_back = 3
trainX, trainY = create_lstm_dataset(train, look_back)
testX, testY = create_lstm_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2, callbacks=[early_stopping], validation_split=0.1)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate mean squared error
trainScore = mean_squared_error(trainY[0], trainPredict[:, 0])
print('Train Mean Squared Error: %.2f' % trainScore)
testScore = mean_squared_error(testY[0], testPredict[:, 0])
print('Test Mean Squared Error: %.2f' % testScore)

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(df[target][s+1:len(train)].index.values, df[target][s+1:len(train)].values, label='Train Data')
plt.plot(df[target][s+1+len(train):].index.values, df[target][s+1+len(train):].values, label='Test Data')
#plt.plot(range(look_back + 1, len(trainPredict) + look_back + 1), trainPredict, label='Training Predictions')
plt.plot(df[target][s+1+len(train)+4:].index.values, testPredict, label='Predictions')



# # Fix the test predictions plot
# test_start = len(trainPredict) + look_back * 2 + 2
# plt.plot(range(test_start, test_start + len(testPredict)), testPredict, label='Test Predictions')

plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.title('LSTM')
plt.legend()
plt.tight_layout()
plt.show()

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
    MSE_train = np.nanmean(e_sq[skip_first_n_obs:n])
    VAR_train = np.nanvar(e[skip_first_n_obs:n])
    MSE_test = np.nanmean(e_sq[n:])
    VAR_test = np.nanvar(e[n:])
    mean_res_train = np.nanmean(e[skip_first_n_obs:n])
    return e, e_sq, MSE_train, VAR_train, MSE_test, VAR_test, mean_res_train





trainY2 = pd.Series(trainY[0])
testY2 = pd.Series(testY[0])
trainPredict2 = pd.Series(trainPredict[:, 0])
testPredict2 = pd.Series(testPredict[:, 0])



pd.concat([trainPredict2, testPredict2])

df_final = pd.DataFrame(list(zip(pd.concat([trainY2, testY2], axis=0), pd.concat([trainPredict2, testPredict2], axis=0))), columns=['y', 'y_pred'])
print(df_final)

e, e_sq, MSE_train, VAR_train, MSE_test, VAR_test, mean_res_train = cal_errors(df_final['y'].to_list(), df_final['y_pred'].to_list(), len(trainY2), 0)

lags=50

import statsmodels.api as sm
method_name = 'LSTM'
q_value = sm.stats.acorr_ljungbox(e[:len(trainY2)], lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
#print(f"Q-Value for training set Method) : ", np.round(Q, 2))
#q_value = toolkit.Cal_q_value(e[:len(y_train)], lags, len(y_train), 2)
print('Error values using {} method'.format(method_name))
print('MSE Prediction data: ', round(MSE_train, 2))
print('MSE Forecasted data: ', round(MSE_test, 2))
print('Variance Prediction data: ', round(VAR_train, 2))
print('Variance Forecasted data: ', round(VAR_test, 2))
print('mean_res_train: ', round(mean_res_train, 2))
print('Q-value: ', round(q_value, 2))
var_f_vs_r = round(VAR_test / VAR_train, 2)
print(f'var(forecast errors)/var(Residual errors): {var_f_vs_r:.2f}')

l_err = [[method_name, MSE_train, MSE_test, VAR_train, VAR_test, mean_res_train, q_value, var_f_vs_r]]
df_err2 = pd.DataFrame(l_err, columns=['method_name', 'MSE_train', 'MSE_test', 'VAR_train', 'VAR_test', 'mean_res_train', 'Q-value', 'Var_test vs Var_train'])
print(df_err2)







train_error = trainY[0] - trainPredict[:, 0]
test_error = testY[0] - testPredict[:, 0]

# Plot forecast errors
plt.figure(figsize=(12, 6))
plt.plot(train_error, label='Training Forecast Error')
plt.plot(range(len(train_error), len(train_error) + len(test_error)), test_error, label='Test Forecast Error')
plt.xlabel('Time')
plt.ylabel('Forecast Error')
plt.legend()
plt.show()





import numpy as np
from scipy.stats import chi2
from sklearn.metrics import r2_score

# Calculate Q value
n = len(test_error)
chi_critical = 5
Q = n * (1 - r2_score(testY[0], testPredict[:, 0]))

# Check if Q is greater than the chi-squared critical value
if Q > chi_critical:
    print(f"Q value ({Q:.2f}) is greater than the chi-squared critical value ({chi_critical}).")
else:
    print(f"Q value ({Q:.2f}) is less than or equal to the chi-squared critical value ({chi_critical}).")

# Calculate MSE
MSE = mean_squared_error(testY[0], testPredict[:, 0])
print(f"MSE: {MSE:.2f}")

# Calculate R2
R2 = r2_score(testY[0], testPredict[:, 0])
print(f"R2: {R2:.2f}")

# Calculate variance for forecast error
var_forecast_error = np.var(test_error)
print(f"Variance for forecast error: {var_forecast_error:.2f}")

# Calculate variance of residual error
residual_error = df[target][s+1:].values[test_start:test_start + len(testPredict)] - testPredict[:, 0]
var_residual_error = np.var(residual_error)
print(f"Variance of residual error: {var_residual_error:.2f}")

# Compare variance of residual error to variance of forecast error
if var_residual_error > var_forecast_error:
    print("Variance of residual error is greater than the variance of forecast error.")
else:
    print("Variance of residual error is less than or equal to the variance of forecast error.")

