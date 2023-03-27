import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import Tool
import numpy as np

url = 'https://raw.githubusercontent.com/sshikha78/Solar-Radiation-Prediction/main/SolarPrediction.csv'
df = pd.read_csv(url,index_col = 'Data')
date = pd.date_range(start = '2016-09-29',
                    periods = len(df),
                    freq='D')
df.index = date
df_hourly = df.resample('H').mean()
df_hourly.to_csv('SolarPrediction_hourly.csv')

print(df.shape)
print(df.info())




