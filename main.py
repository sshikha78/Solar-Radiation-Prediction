import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
import Tool
import numpy as np
import seaborn as sns
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Loading DATA

pd.set_option('display.max_columns', None)
url = 'https://raw.githubusercontent.com/sshikha78/Solar-Radiation-Prediction/main/SolarPrediction.csv'
df = pd.read_csv(url)

df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Time'])
df.sort_values(by='Datetime', inplace=True)
df = df.drop(['UNIXTime'], axis=1)
print(df)
print(df.info())
df['TimeSunRise'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunRise']).astype(np.int64)
df['TimeSunSet'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunSet']).astype(np.int64)
df_hourly = df.resample('H', on='Datetime').mean()

# Plotting dependent variable ####
Tool.plot_graph(list(df_hourly.index.values), df_hourly['Radiation'], 'Time', 'Radiation', 'Radiation over Time')

#   Check NA value    #####

print(f'NA value: \n{df_hourly.isna().sum()}')
df_hourly.reset_index(inplace=True)
print(df_hourly)
print(df_hourly[df_hourly.isna().any(axis=1)])
col_list = df_hourly.columns[df_hourly.isna().any()].tolist()
index_list = df_hourly[df_hourly.isna().any(axis=1)].index.tolist()
# Filling NA with forecasted value of drift method  #
for col in col_list:
    for index in index_list:
        y_pred = Tool.drift(df_hourly[col], index)
        df_hourly[col][index] = y_pred[index]


# Checking if NA remain     ####
print(f'NA value: \n{df_hourly.isna().sum()}')
df_hourly.set_index('Datetime', inplace=True)
print(df_hourly[170:190])

# Plotting dependent variable vs time   ####
Tool.plot_graph(list(df_hourly.index.values), df_hourly['Radiation'], 'Time', 'Radiation', 'Radiation over Time')


df_hourly['TimeSunRise'] = pd.to_datetime(df_hourly['TimeSunRise'], format='%Y-%m-%d %H:%M:%S')
df_hourly['TimeSunSet'] = pd.to_datetime(df_hourly['TimeSunSet'], format='%Y-%m-%d %H:%M:%S')
date = pd.date_range(start = '2016-09-01', periods = len(df_hourly), freq='H')
df_hourly.index = date
print("NaN values  interpolation:", df_hourly.isna().sum())
print(df_hourly.describe())
print(df_hourly.shape)


#   ACF/PACF Plot    ####
Tool.ACF_PACF_Plot(df_hourly['Radiation'], lags=24, method_name='Solar Radiation')


#   Correlation Matrix   ####

plt.figure(figsize=(16, 8))
corr_matrix = df_hourly.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

#   DATA SPLITING 	    ####

df_train, df_test= train_test_split(df_hourly, test_size=0.2, shuffle=False)

#  Stationarity check   ####

# Perform Rolling mean and variance,ADF and KPSS tests on the raw data
Tool.Graph_rolling_mean_var(df_hourly, col='Radiation')
print('---ADF Test for Radiation---')
Tool.adf_Cal_pass(df_hourly['Radiation'], "Radiation")
print('---KPSS Test for Radiation---')
Tool.kpss_test(df_hourly['Radiation'])
Tool.ACF_PACF_Plot(df_hourly['Radiation'],50,"Without differncing")

# Seasonal Differencing
s = 24
df_hourly['seasonal_d_o_1'] = Tool.seasonal_differencing(df_hourly['Radiation'], seasons=s)
Tool.plot_graph(list(df_hourly.index.values), df_hourly['seasonal_d_o_1'], 'Time', 'seasonal_d_o_1', 'Radiation over Time')


# Stationarity on seasonaly differenced data

Tool.ACF_PACF_Plot(df_hourly['seasonal_d_o_1'][s:], lags=60,method_name='Seasonal Differncing')
print('---ADF Test for Seasonal Diff 1 Radiation---')
Tool.adf_Cal_pass(df_hourly['seasonal_d_o_1'][s:], "Seasonal Diff 1 Radiation")
print('---KPSS Test for Seasonal Diff 1 Radiation---')
Tool.kpss_test(df_hourly['seasonal_d_o_1'][s:])
Tool.Graph_rolling_mean_var(df_hourly[s:], 'seasonal_d_o_1')

# Doing a non-seasonal differencing after the seasonal differrencing
# Transforming data to make it stationary
df_hourly['diff_order_1'] = Tool.non_seasonal_differencing(df_hourly['seasonal_d_o_1'], s)

# Plotting dependent variable vs time
Tool.plot_graph(list(df_hourly.index.values), df_hourly['diff_order_1'], 'Time', 'diff_order_1', 'Radiation over Time')

# Stationarity Tests on transformed data
Tool.ACF_PACF_Plot(df_hourly['diff_order_1'][s+1:], lags=60,method_name='Differncing Order-1')
print('---ADF Test for  Radiation diff---')
Tool.adf_Cal_pass(df_hourly['diff_order_1'][s+1:], "diff_order_1 Radiation")
print('---KPSS Test for  Radiation diff---')
Tool.kpss_test(df_hourly['diff_order_1'][s+1:])
Tool.Graph_rolling_mean_var(df_hourly[s+1:], 'diff_order_1')

# STL Decomposition

Tool.stl_decomposition(df_hourly['Radiation'], 'Radiation')
Tool.stl_decomposition(df_hourly['diff_order_1'][s+1:], 'diff_order_1')


# # HOLT WINTERS METHOD	    ####

df1 = df_hourly.copy()
x = df1[s+1:].drop(['Radiation','Datetime','diff_order_1','Data','Time'], axis=1)
y = df1[s+1:]['diff_order_1']
x_train1, x_test1,y_train1, y_test1 = train_test_split(x,y, shuffle=False, test_size=0.20)
Tool.holt_winters_forecast(df_train['Radiation'].values, df_test['Radiation'].values)

# #	FEATURE SELECTION	####

H = np.dot(x_train1.T ,x_train1)
U, S, V = np.linalg.svd(H, full_matrices=True)
print("Singular values:", S)
cond_num = np.linalg.cond(x_train1)
print("Condition number:", cond_num)