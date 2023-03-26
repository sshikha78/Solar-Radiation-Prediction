import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from scipy import signal


def adf_Cal_pass(y,name):
    result = adfuller(y)
    print(f"ADF TEST - {name}: ")
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    print("Results of KPSS Test - ")
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'% key] = value
    print(kpss_output)


def Graph_rolling_mean_var(data, col=None):
    pass_rolling_mean_list = []
    pass_rolling_var_list = []

    for i in range(len(data)):

        subset1 = data.head(i)

        pass_rolling_mean = subset1[f'{col}'].mean()
        pass_rolling_var = np.var(subset1[f'{col}'])

        pass_rolling_mean_list.append(pass_rolling_mean)
        pass_rolling_var_list.append(pass_rolling_var)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(pass_rolling_mean_list)
    ax[0].set_title('Rolling Mean - {}'.format(col))
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Magnitude")
    ax[1].plot(pass_rolling_var_list, label='Varying Variance')
    ax[1].set_title('Rolling Variance - {}'.format(col))
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Magnitude")
    plt.legend(loc='lower right')
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def pass_cal_rolling_mean_var(data, col):
    pass_rolling_mean_list = []
    pass_rolling_var_list = []

    for i in range(len(data)):

        subset1 = data.head(i)

        pass_rolling_mean = subset1[f'{col}'].mean()
        pass_rolling_var = np.var(subset1[f'{col}'])

        pass_rolling_mean_list.append(pass_rolling_mean)
        pass_rolling_var_list.append(pass_rolling_var)

    return pass_rolling_mean_list, pass_rolling_var_list

def non_seasonal_differencing(column, n):
    difference = []
    for i in range(0, n):
        difference.append(np.nan)
    for i in range(n, len(column)):
        difference.append(column[i] - column[i-1])
    return difference

#Autocorrelation Calculator
def Auto_corr_cal(y,lags):
    mean = np.mean(y)
    var = np.var(y)
    num = 0
    den = 0
    ry = []
    for i in range(0, len(y)):
        den += (y[i] - mean ) ** 2
    for lag in range(0, lags+1):
        num=0
        for i in range(lag, len(y)):
            num += (y[i] - mean) * (y[i - lag] - mean)
        ry.append(num / den)

    return ry

def Auto_corr_plot(y,lags,method_name=None):
    y = np.array(y)
    ry = Auto_corr_cal(y,lags)
    # print(ry)
    final = []
    final.extend(ry[:0:-1])
    final.extend(ry)
    lag_f = []
    lags = list(range(0, lags + 1, 1))
    lag_f.extend(lags[:0:-1])
    lag_f = [value * (-1) for value in lag_f]
    lag_f.extend(lags)
    markers, stemlines, baseline = plt.stem(lag_f, final)
    plt.setp(markers, color='red', marker='o')
    plt.axhspan((-1.96 / np.sqrt(len(y))), (1.96 / np.sqrt(len(y))), alpha=0.2, color='blue')
    plt.xlabel('LagS')
    plt.ylabel('Magnitude')
    plt.title(f'AutoCorrelation of {method_name}')
    plt.show()



yt_pred = []
error = []
e_squared=[]
def average_method(t,yt,n):
    for i in range(0, len(yt)):
        if i  == 0:
            yt_pred.append(np.nan)
        elif i < n :
            yt_pred.append(sum(yt[:i]) / i)
        else:
            yt_pred.append(sum(yt[:n]) / (n))
    return yt_pred


def error_method(yt, pred, n,s_o):
    error=[]
    e_squared=[]
    for i in range(0, len(yt)):
        if i == 0:
            error.append(np.nan)
            e_squared.append(np.nan)
        else:
            error.append(yt[i] - pred[i])
            e_squared.append(error[i] ** 2)

    mse_tr = sum(e_squared[s_o:n]) / len(e_squared[s_o:n])
    mse_ts = sum(e_squared[n:]) / len(e_squared[n:])
    res_mean = np.nanmean(error[s_o:n])
    var_pred = np.nanvar(error[s_o:n])
    var_fcst = np.nanvar(error[n:])
    return error, e_squared, mse_tr, mse_ts,var_pred,var_fcst,res_mean

def plot_forecast(t, yt,method_n, yt_pred, n):
    y_tr = yt[:n]
    y_ts = yt[n:]
    plt.plot(t[:n], y_tr, marker='o', color='blue', label='Training Set')
    plt.plot(t[n:], y_ts, marker='s', color='green', label='Test Set')
    plt.plot(t[n:], yt_pred[n:], marker='^', color='red', label='Step Forecast')
    # plt.plot([t[-1] + 1, t[-1] + 2, t[-1] + 3], yt_pred, marker='o', color='green', label='Forecast')
    plt.title(method_n)
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()

def cal_q_value(error, lags, n, s_o=0):
    acf = 0
    fd = [x for x in error[s_o:] if np.isnan(x) == False]
    acf_value = Auto_corr_cal(fd, lags)
    acf_value_sq = [i ** 2 for i in acf_value[1:]]
    acf = sum(acf_value_sq)
    q_value = len(fd) * (acf)
    return q_value

def naive_method(yt, n,h):
    yt_pred_n = []
    for i in range(len(yt)):
        if i == 0:
            yt_pred_n.append(np.nan)
        elif(i<n):
            yt_pred_n.append(yt[i - 1])
        else:
            train = yt[:n]
            yt_pred_n.append(train[-1])
    return yt_pred_n

def drift_method(t,yt,n):
    yt_pred_d=[]
    for i in range(len(t)):
        if i < 2:
            yt_pred_d.append(np.nan)
        elif i < n:
            h = 1
            yd = yt[i-1] + (h * (yt[i-1] - yt[0])) / (i-1)
            yt_pred_d.append(yd)
        else:
            h = i - n + 1
            yd = yt[n-1] + (h * (yt[n-1] - yt[0])) / (n-1)
            yt_pred_d.append(yd)
    return yt_pred_d

def ses_method(t,yt,n,alpha):
    l0=yt[0]
    yt_pred_s=[np.nan]
    yt_pred_s.append(alpha * l0 + (1 - alpha) * l0)
    for i in range(2, len(yt)):
        if i<n:
            yt_pred_s.append(alpha * yt[i-1] + (1 - alpha) * yt_pred_s[i - 1])
        else:
            yt_pred_s.append(alpha * yt[n - 1] + (1 - alpha) * yt_pred_s[n - 1])
    return yt_pred_s



def backward_step_Reg(X, y):
    final_fea_list = []
    model_final = {}
    model = sm.OLS(y, X).fit()
    model_final['Overall-Model'] = [model.aic, model.bic, model.rsquared_adj]
    while len(X.columns) > 1:
        scores = {}
        for col in X.columns:
            X_temp = X.drop(col, axis=1)
            model_temp = sm.OLS(y, X_temp).fit()
            scores[col] = [model_temp.aic, model_temp.bic, model_temp.rsquared_adj]
        feature_best = min(scores, key=lambda k: (scores[k][0], scores[k][1], -scores[k][2]))
        if scores[feature_best][0] < model.aic and scores[feature_best][1] < model.bic and scores[feature_best][
            2] > model.rsquared_adj:
            final_fea_list.append(feature_best)
            model = sm.OLS(y, X.drop(feature_best, axis=1)).fit()
            model_final[feature_best] = scores[feature_best]
            X = X.drop(feature_best, axis=1)
        else:
            break
    return model_final, final_fea_list



def moving_average(df, m, fold):
    y = df.values.flatten()
    k = int((m-1)/2)
    ma = np.full(y.size, np.nan)
    f_ma = np.full(y.size, np.nan)

    if m % 2 != 0:
        for i in range(k, y.size-k):
            ma[i] = (1/m)*np.sum(y[i-k:i+k+1])
        df_ma = pd.DataFrame(list(zip(df['Temp'], ma)), index=df.index, columns=['Temp',f'{m}-MA'])
    else:
        for i in range(k, y.size-k-1):
            ma[i] = (1/m)*np.sum(y[i-k:i+k+1+1])
        if fold > 0:
            for i in range(k+1, y.size-k-1):
                f_ma[i] = (1/fold)*np.sum(ma[i-1:i+1])
            df_ma = pd.DataFrame(list(zip(df['Temp'], f_ma)), index=df.index, columns=['Temp',f'2x{m}-MA'])
        else:
            df_ma = pd.DataFrame(list(zip(df['Temp'], ma)), index=df.index, columns=['Temp',f'{m}-MA'])

    return df_ma



def AR2_process(e,N):
    np.random.seed(6313)
    y = np.zeros(len(e))
    for t in range(N):
        if t == 0:
            y[t] = e[t]
        elif t == 1:
            y[t] = e[t] + 0.5 * y[t - 1]
        else:
            y[t] = e[t] + 0.5 * y[t - 1] + 0.2 * y[t - 2]
    np.set_printoptions(precision=2)
    return y

def dlsim_AR2(e,num,den):
    system = (num, den, 1)
    t, y_dlsim = signal.dlsim(system, e)
    print(f"y dlsim_AR2- {y_dlsim[:5].flatten()}")
    return y_dlsim.flatten()

def AR2_lse(e,c1,c2,n):
    np.random.seed(6313)
    y = np.zeros(n)
    for i in range(2, n):
        y[i] = c1 * y[i - 1] + c2 * y[i - 2] + e[i]
    X = np.column_stack((y[1:n - 1], y[0:n - 2]))
    Y = y[2:]
    a_hat = -(np.linalg.inv(X.T @ X) @ X.T @ Y)
    return a_hat

def MA2_Process(n, c1, c2):
    np.random.seed(6313)
    e = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(len(e)):
        if t == 0:
            y[t] = e[t]
        elif t == 1:
            y[t] = e[t] + c1 * e[t - 1]
        else:
            y[t] = e[t] + c1 * e[t - 1] + c2 * e[t - 2]
    return y, e


def dlsim_MA2(e,num,den):
    system = (num, den, 1)
    t, y_dlsim = signal.dlsim(system, e)
    print(f"y dlsim MA2_method - {y_dlsim[:5].flatten()}")
    return y_dlsim.flatten()