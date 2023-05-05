import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
import Tool
import numpy as np
import seaborn as sns

# display all the columns
pd.set_option('display.max_columns', None)

url = 'https://raw.githubusercontent.com/sshikha78/Solar-Radiation-Prediction/main/SolarPrediction.csv'
df = pd.read_csv(url)

# Convert the "Data" and "Time" columns to a datetime data type
df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Time'])
df.sort_values(by='Datetime', inplace=True)

# Remove the "UNIXTime" column
df = df.drop(['UNIXTime'], axis=1)

print(df)
print(df.info())

# Convert the "TimeSunRise" and "TimeSunSet" columns to int64 data type
df['TimeSunRise'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunRise']).astype(np.int64)
df['TimeSunSet'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunSet']).astype(np.int64)


# Resample the DataFrame by hour and calculate the mean of each column
df_hourly = df.resample('H', on='Datetime').mean()

# Plotting dependent variable vs time
plt.figure(figsize=(16, 8))
plt.plot(list(df_hourly.index.values), df_hourly['Radiation'])
plt.xlabel('Time')
plt.ylabel('Radiation')
plt.title('Radiation over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Check NA value
print(f'NA value: \n{df_hourly.isna().sum()}')

# Reetting index for getting the integer index values of NA records
df_hourly.reset_index(inplace=True)
print(df_hourly)


# Check for NA in any column
print(df_hourly[df_hourly.isna().any(axis=1)])

# Retrieving columns containing NA valuess
col_list = df_hourly.columns[df_hourly.isna().any()].tolist()
# Retrieving index number of rows containing NA valuess
index_list = df_hourly[df_hourly.isna().any(axis=1)].index.tolist()

# Filling na with forecasted value of drift method
for col in col_list:
    for index in index_list:
        y_pred = Tool.drift(df_hourly[col], index)
        df_hourly[col][index] = y_pred[index]


# Checking if NA remain
print(f'NA value: \n{df_hourly.isna().sum()}')

# Setting the Datetime as index again
df_hourly.set_index('Datetime', inplace=True)

print(df_hourly[170:190])

# Plotting dependent variable vs time
plt.figure(figsize=(16, 8))
plt.plot(list(df_hourly.index.values), df_hourly['Radiation'])
plt.xlabel('Time')
plt.ylabel('Radiation')
plt.title('Radiation over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Convert the "TimeSunRise" and "TimeSunSet" columns back to datetime data type
df_hourly['TimeSunRise'] = pd.to_datetime(df_hourly['TimeSunRise'], format='%Y-%m-%d %H:%M:%S')
df_hourly['TimeSunSet'] = pd.to_datetime(df_hourly['TimeSunSet'], format='%Y-%m-%d %H:%M:%S')

# Set the index of the DataFrame to a date range
date = pd.date_range(start = '2016-09-01', periods = len(df_hourly), freq='H')
df_hourly.index = date


# Check for NaN values again
print("NaN values  interpolation:", df_hourly.isna().sum())

# Summarizing the dataset
print(df_hourly.describe())

# Shape of  the dataset
print(df_hourly.shape)


# ACF/PACF of the dependent variable
plt.figure(figsize=(12, 8))
Tool.Auto_corr_plot(df_hourly['Radiation'], lags=24, method_name='Solar Radiation')
plt.show()

# Correlation Matrix
plt.figure(figsize=(16, 8))
corr_matrix = df_hourly.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Matrix')
plt.show()

# Histogram
plt.rcParams["font.size"] = 14
df_hourly.hist(figsize=(12,12))
plt.title('Histogram')
plt.tight_layout()
plt.show()

# Split dataset into train and test sets
X = df_hourly.drop(['Radiation'], axis=1)
y = df_hourly['Radiation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set size: {len(X_train)} rows and {len(X_train.columns)+1} columns')
print(f'Testing set size: {len(X_test)} rows and {len(X_test.columns)+1} columns')

#  Stationarity check

#Perform Rolling mean and variance,ADF and KPSS tests on the raw data
Tool.Graph_rolling_mean_var(df_hourly, col='Radiation')
Tool.adf_Cal_pass(df_hourly['Radiation'], "Radiation")
Tool.kpss_test(df_hourly['Radiation'])
Tool.ACF_PACF_Plot(df_hourly['Radiation'],50,"Wirhout differncing")


# Seasonal Differencing
s = 24
df_hourly['seasonal_d_o_1'] = Tool.seasonal_differencing(df_hourly['Radiation'], seasons=s)
#print(df[['pollution', 'seasonal_d_o_1']].head(60))

# Plotting dependent variable vs time
plt.figure(figsize=(16, 8))
plt.plot(list(df_hourly.index.values), df_hourly['seasonal_d_o_1'])
plt.xlabel('Time')
plt.ylabel('seasonal_d_o_1')
plt.title('Pollution over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Stationarity on seasonaly differenced data

Tool.ACF_PACF_Plot(df_hourly['seasonal_d_o_1'][s:], lags=60)
Tool.adf_Cal_pass(df_hourly['seasonal_d_o_1'][s:], "Seasonal Diff 1 Radiation")
Tool.kpss_test(df_hourly['seasonal_d_o_1'][s:])
Tool.Graph_rolling_mean_var(df_hourly[s:], 'seasonal_d_o_1')

# Doing a non-seasonal differencing after the seasonal differrencing
# Transforming data to make it stationary
df_hourly['diff_order_1'] = Tool.non_seasonal_differencing(df_hourly['seasonal_d_o_1'], s)

# Plotting dependent variable vs time
plt.figure(figsize=(16, 8))
plt.plot(list(df_hourly.index.values), df_hourly['diff_order_1'])
plt.xlabel('Time')
plt.ylabel('diff_order_1')
plt.title('Pollution over Time')
plt.legend()
plt.tight_layout()
plt.show()

# Stationarity Tests on transformed data
Tool.ACF_PACF_Plot(df_hourly['diff_order_1'][s+1:], lags=60)
Tool.adf_Cal_pass(df_hourly['diff_order_1'][s+1:], "diff_order_1 Radiation")
Tool.kpss_test(df_hourly['diff_order_1'][s+1:])
Tool.Graph_rolling_mean_var(df_hourly[s+1:], 'diff_order_1')

# STL Decomposition
radiation = pd.Series(df_hourly['Radiation'].values,index = date,
                 name = 'Radiation')

STL = STL(radiation, period=24)
res = STL.fit()
plt.figure(figsize=(16,10))
fig = res.plot()
plt.grid()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid
plt.figure()

plt.figure(figsize=(16,10))
plt.plot(df_hourly.index, T.values, label = 'trend')
plt.plot(df_hourly.index, S.values, label = 'Seasonal')
plt.plot(df_hourly.index, R.values, label = 'residuals')

plt.show()
var_resi1 = np.var(R)
var_resid_trend = np.var(T + R)
Ft = np.max([0, 1 - var_resi1 / var_resid_trend])
print("The strength of trend for this data set is  ", Ft)

var_resi = np.var(R)
var_resid_seasonal = np.var(S + R)
St = np.max([0, 1 - var_resi / var_resid_seasonal])
print("The strength of seasonality for this data set is  ", St)
