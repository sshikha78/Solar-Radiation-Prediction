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

# Replace NaN values with the mean of each column
df_hourly = df_hourly.fillna(df_hourly.mean())

# Convert the "TimeSunRise" and "TimeSunSet" columns back to datetime data type
df_hourly['TimeSunRise'] = pd.to_datetime(df_hourly['TimeSunRise'], format='%Y-%m-%d %H:%M:%S')
df_hourly['TimeSunSet'] = pd.to_datetime(df_hourly['TimeSunSet'], format='%Y-%m-%d %H:%M:%S')

# Set the index of the DataFrame to a date range
date = pd.date_range(start = '2016-09-01', periods = len(df_hourly), freq='H')
df_hourly.index = date

# Check for NaN values again
print("NaN values:", df_hourly.isna().sum())

# Summarizing the dataset
print(df_hourly.describe())

# Shape of  the dataset
print(df_hourly.shape)

# Plot of the dependent variable versus time
plt.figure(figsize=(12, 8))
plt.plot(df_hourly.index, df_hourly['Radiation'])
plt.title('Solar Radiation over Time')
plt.xlabel('Time')
plt.ylabel('Solar Radiation (W/m²)')
plt.show()

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


#  Stationarity check

#Perform Rolling mean and variance,ADF and KPSS tests on the raw data
Tool.Graph_rolling_mean_var(df_hourly, col='Radiation')
Tool.adf_Cal_pass(df_hourly['Radiation'], "Radiation")
Tool.kpss_test(df_hourly['Radiation'])

# Perform Rolling mean and variance,ADF and KPSS tests on the transformed data
df_hourly['transformed_data_1'] = Tool.non_seasonal_differencing(df_hourly['Radiation'], 1)
Tool.Graph_rolling_mean_var(df_hourly[1:], 'transformed_data_1')
Tool.adf_Cal_pass(df_hourly['transformed_data_1'][1:], 'transformed_data_1')
Tool.kpss_test(df_hourly['transformed_data_1'][1:])
Tool.Auto_corr_plot(df_hourly['transformed_data_1'][1:], lags=24, method_name='Transformed 1-Radiation')

df_hourly['transformed_data_2'] = Tool.non_seasonal_differencing(df_hourly['transformed_data_1'], 2)
Tool.Graph_rolling_mean_var(df_hourly[2:], 'transformed_data_2')
Tool.adf_Cal_pass(df_hourly['transformed_data_2'][2:],'transformed_data_2')
Tool.kpss_test(df_hourly['transformed_data_2'][2:])
Tool.Auto_corr_plot(df_hourly['transformed_data_2'][2:], lags=24, method_name='Transformed 2-Radiation')

df_hourly['transformed_data_3'] = Tool.non_seasonal_differencing(df_hourly['transformed_data_2'], 2)
Tool.Graph_rolling_mean_var(df_hourly[3:], 'transformed_data_3')
Tool.adf_Cal_pass(df_hourly['transformed_data_3'][3:],'transformed_data_3')
Tool.kpss_test(df_hourly['transformed_data_3'][3:])
Tool.Auto_corr_plot(df_hourly['transformed_data_3'][3:], lags=24, method_name='Transformed 3-Radiation')


df_hourly['df_log'] = np.log(df_hourly['Radiation'])
df_hourly['transformed_data_log'] = Tool.non_seasonal_differencing(df_hourly['df_log'],1)
Tool.Graph_rolling_mean_var(df_hourly[1:], 'transformed_data_log')
Tool.adf_Cal_pass(df_hourly['transformed_data_log'][1:], 'transformed_data_log')
Tool.kpss_test(df_hourly['transformed_data_log'][1:])
Tool.Auto_corr_plot(df_hourly['transformed_data_log'][1:], lags=24, method_name='Transformed log-Radiation')

