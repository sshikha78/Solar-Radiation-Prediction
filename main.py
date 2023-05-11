import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import statsmodels.api as sm
import Tool
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

#==============================================
# Loading DATA ####
#===============================================

pd.set_option('display.max_columns', None)
url = 'https://raw.githubusercontent.com/sshikha78/Solar-Radiation-Prediction/main/SolarPrediction.csv'
df = pd.read_csv(url)
print(df.head().to_string())
df['Datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Time'])
df.sort_values(by='Datetime', inplace=True)
df = df.drop(['UNIXTime'], axis=1)
print(df.head().to_string())
print(df.info())
df['TimeSunRise'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunRise']).astype(np.int64)
df['TimeSunSet'] = pd.to_datetime(df['Data'] + ' ' + df['TimeSunSet']).astype(np.int64)
df_hourly = df.resample('H', on='Datetime').mean()


#==============================================
# Plotting dependent variable ####
#===============================================

Tool.plot_graph(list(df_hourly.index.values), df_hourly['Radiation'], 'Time', 'Radiation', 'Radiation over Time')


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
        y_pred = Tool.drift(df_hourly[col], index)
        df_hourly[col][index] = y_pred[index]


#==============================================
# Checking if NA remain AFTER DRIFT      ####
#===============================================

print(f'NA value: \n{df_hourly.isna().sum()}')
df_hourly.set_index('Datetime', inplace=True)


#==============================================
# Plotting dependent variable vs time   ####
#===============================================

Tool.plot_graph(list(df_hourly.index.values), df_hourly['Radiation'], 'Time', 'Radiation', 'Radiation over Time')

df_hourly['TimeSunRise'] = pd.to_datetime(df_hourly['TimeSunRise'], format='%Y-%m-%d %H:%M:%S')
df_hourly['TimeSunSet'] = pd.to_datetime(df_hourly['TimeSunSet'], format='%Y-%m-%d %H:%M:%S')
date = pd.date_range(start = '2016-09-01', periods = len(df_hourly), freq='H')
df_hourly.index = date
print("NaN values  interpolation:", df_hourly.isna().sum())
print(df_hourly.describe().to_string())
print(df_hourly.info())
print(df_hourly.shape)

#==============================================
#   ACF/PACF Plot    ####
#===============================================

Tool.ACF_PACF_Plot(df_hourly['Radiation'], lags=50, method_name='Solar Radiation')


#==============================================
#  Correlation Matrix   ####
#===============================================

plt.figure(figsize=(16, 8))
corr_matrix = df_hourly.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

#==============================================
#  Stationarity check   ####
#===============================================

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
Tool.plot_graph(list(df_hourly.index.values), df_hourly['diff_order_1'], 'Time', 'diff_order_1', 'Radiation over Time')
Tool.ACF_PACF_Plot(df_hourly['diff_order_1'][s+1:], lags=60,method_name='Differncing Order-1')
print('---ADF Test for  Radiation diff---')
Tool.adf_Cal_pass(df_hourly['diff_order_1'][s+1:], "diff_order_1 Radiation")
print('---KPSS Test for  Radiation diff---')
Tool.kpss_test(df_hourly['diff_order_1'][s+1:])
Tool.Graph_rolling_mean_var(df_hourly[s+1:], 'diff_order_1')

#==============================================
# STL Decomposition
#===============================================

Tool.stl_decomposition(df_hourly['Radiation'], 'Radiation')
Tool.stl_decomposition(df_hourly['diff_order_1'][s+1:], 'diff_order_1')


#==============================================
# #	FEATURE SELECTION	####
#===============================================

df1 = df_hourly.copy()
print(df1.info())
x = df1[s+1:].drop(['Radiation','TimeSunRise','diff_order_1','TimeSunSet','seasonal_d_o_1'], axis=1)
y = df1[s+1:]['diff_order_1']

x_train1, x_test1,y_train1, y_test1 = train_test_split(x,y, shuffle=False, test_size=0.20)

col_list = x_train1.columns

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train1)
X_test = scaler.transform(x_test1)

X_train = pd.DataFrame(X_train, columns=col_list)
X_test = pd.DataFrame(X_test, columns=col_list)

H = np.dot(X_train.T ,X_train)
U, S, V = np.linalg.svd(H, full_matrices=True)
print("Singular values:", S)
cond_num = np.linalg.cond(X_train)
print("Condition number:", cond_num)

vif = pd.DataFrame()
vif["Feature"] = X_train.columns
vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print(vif)

X_train_2 = sm.add_constant(X_train, prepend=True)
X_test_2 = sm.add_constant(X_test, prepend=True)
model = sm.OLS(list(y_train1), X_train_2)
results = model.fit()
print(results.summary())

X_train_2.drop(['WindDirection(Degrees)'], axis=1, inplace=True)
model_Final = sm.OLS(list(y_train1), X_train_2).fit()
print(model_Final.summary())

X_train_2.drop(['Pressure'], axis=1, inplace=True)
model_Final = sm.OLS(list(y_train1), X_train_2).fit()
print(model_Final.summary())

X_train_2.drop(['Temperature'], axis=1, inplace=True)
model_Final = sm.OLS(list(y_train1), X_train_2).fit()
print(model_Final.summary())

X_train_2.drop(['Humidity'], axis=1, inplace=True)
model_Final = sm.OLS(list(y_train1), X_train_2).fit()
print(model_Final.summary())



#==============================================
#   DATA SPLITTING 	    ####
#===============================================
index_df_hourly = df_hourly.index
df_train, df_test = train_test_split(df_hourly, test_size=0.2, shuffle=False)


#==============================================
# # Base Model	    ####
#===============================================

print('-----AVERAGE METHOD----------')
Tool.forecast_method(df_hourly[s+1:]['diff_order_1'], len(df_train), 'Average')
print('-----NAIVE METHOD----------')
Tool.forecast_method(df_hourly[s+1:]['diff_order_1'], len(df_train), 'Naive')
print('-----DRIFT METHOD----------')
Tool.forecast_method(df_hourly[s+1:]['diff_order_1'], len(df_train), 'Drift')
print('-----SES METHOD----------')
Tool.forecast_method(df_hourly[s+1:]['diff_order_1'], len(df_train), 'SES')
print('-----HOLT WINTER METHOD----------')
Tool.forecast_method(df_hourly[s+1:]['diff_order_1'], len(df_train), 'Holt-Winters')


#==============================================
#  Multiple Linear regression Model  #####
#===============================================

X_train = sm.add_constant(X_train, prepend=True)
X_test = sm.add_constant(X_test, prepend=True)
model_full_final = sm.OLS(list(y_train1), X_train).fit()
print(model_full_final.summary())

y_pred = model_full_final.predict(X_train)
X_test = X_test[X_train.columns.to_list()]
y_forecast = model_full_final.predict(X_test)


df_ols = pd.DataFrame(list(zip(pd.concat([y_train1, y_test1], axis=0), pd.concat([y_pred, y_forecast], axis=0))), columns=['y', 'y_pred'], index = df_hourly[s+1:].index)
plt.figure(figsize=(16,8))
plt.plot(df_ols[:len(y_train1)].index, y_train1, label="Train")
plt.plot(df_ols[len(y_train1):].index, y_test1, label="Test")
plt.plot(df_ols[len(y_train1):].index, y_forecast, label="Forecast")
plt.xlabel("Time")
plt.ylabel("Radiation")
plt.title("Forcast using OLS")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

df_final = pd.DataFrame(list(zip(pd.concat([y_train1, y_test1], axis=0), pd.concat([y_pred, y_forecast], axis=0))), columns=['y', 'y_pred'])
e, e_sq, mse_tr, var_pred, mse_ts, var_fcst, res_mean = Tool.error_method(df_final['y'].to_list(), df_final['y_pred'].to_list(), len(y_train1), 0)
Q = sm.stats.acorr_ljungbox(model_full_final.resid, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
print(f"Q-Value for training set Method) : ", np.round(Q, 2))
var_f_vs_r = round((var_fcst) / (var_pred), 2)
print(f'var(forecast errors)/var(Residual errors): {var_f_vs_r:.2f}')
Tool.Auto_corr_plot(model_full_final.resid, lags=20, method_name='ACF Plot for errors Prediction Errors')

ols_data = ["Multi-Linear Regression", mse_tr, mse_ts, var_fcst, res_mean, np.round(Q, 2),var_f_vs_r]

print('T-Test')
print(model_full_final.pvalues)
print('\nF-Test')
print(model_full_final.f_pvalue)



#==============================================
# GPAC ####
#===============================================

lags = 100
ry = sm.tsa.stattools.acf(y_train1, nlags=lags)
# Tool.calc_GPAC(ry, J=50, K=50, savepath=f'gpac.png')
Tool.Gpac(ry,j_max=26,k_max=26)

#==============================================
# SARIMA    ####
#===============================================

sarimax = Tool.ARIMA_method(0, 0, 1, 24, y_train1, y_test1)
Tool.ARIMA_method(1, 0, 0, 24, y_train1, y_test1)

#==============================================
# Levenberg Marquardt algorithm  ####
#===============================================


Tool.lm(y_train1,0,1)

# Tool.lm(y_train1,1,0)



#==============================================
#  Table ####
#===============================================

from tabulate import tabulate
def forecast_and_table(y, n, models, alpha=0.5, lags=20):
    results = []
    for model in models:
        y_pred, e, e2, mse_tr, mse_ts, var_pred, var_fcst, res_mean, var_f_vs_r = Tool.Base_model(y, n, model, alpha, lags)
        Q = sm.stats.acorr_ljungbox(e[2:n], lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
        results.append([model, mse_tr, mse_ts, var_fcst, res_mean, np.round(Q, 2), var_f_vs_r])
    return results
models = ['Average', 'Naive', 'Drift', 'SES','Holt-Winters']
results = forecast_and_table(df_hourly[s+1:]['diff_order_1'], len(df_train), models)
results.append(ols_data)
results.append(sarimax)
lstm_error = ['LSTM', 13081.633729, 9423.703469, 9131.291846,  -16.878022, 736.416059, 0.71]
results.append(lstm_error)
headers = ["Method", "MSE Train", "MSE Test", "Var Forecast", "Mean Residual Train", "Q Value","Ratio Vres_er vs V_for_err "]
print(tabulate(results, headers=headers))



