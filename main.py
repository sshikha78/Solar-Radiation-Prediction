import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import Tool
import numpy as np

# display all the columns
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
# Re
df_hourly = df.resample('H', on='Datetime').mean()
# Replace NaN values with the mean of each column
df_hourly = df_hourly.fillna(df_hourly.mean())


df_hourly['TimeSunRise'] = pd.to_datetime(df_hourly['TimeSunRise'], format='%Y-%m-%d %H:%M:%S')
df_hourly['TimeSunSet'] = pd.to_datetime(df_hourly['TimeSunSet'], format='%Y-%m-%d %H:%M:%S')
date = pd.date_range(start = '2016-09-01', periods = len(df_hourly), freq='H')
df_hourly.index = date


# Check for NaN values again
print("NaN values:", df_hourly.isna().sum())

# Summarizing the dataset
print(df_hourly.describe())

# Shape of  the dataset
print(df_hourly.shape)

plt.figure(figsize=(16,8))
plt.plot(list(df_hourly.index.values), df_hourly.Radiation.values)
plt.show()

plt.rcParams["font.size"] = 14
df_hourly.hist(figsize=(12,12))
plt.tight_layout()
plt.show()

# fill na with mean before datetime conversion on line 23

# Feature Selection

# Correlation