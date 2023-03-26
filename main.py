import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
import Tool
import numpy as np
url = ''
df = pd.read_csv(url, index_col='Date',)
date = pd.date_range(start = '1981-01-01',
                    periods = len(df),
                    freq='D')
