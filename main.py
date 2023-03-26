import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import Tool
import numpy as np

url = 'https://raw.githubusercontent.com/sshikha78/Solar-Radiation-Prediction/main/SolarPrediction.csv'
df = pd.read_csv(url)
print(df)


