import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import seaborn as sns

np.random.seed(6313)

def adf_Cal_pass(x):
    result = adfuller(x)
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

def Cal_Rolling_MeanVar(dataset, column_name):
    for i in range(1, len(dataset)):
        return np.mean(dataset[column_name].head(i)), np.var(dataset[column_name].head(i))

def CalRollingMeanVarGraph(dataset, column_name):
    df_plot = pd.DataFrame(columns=['Samples', 'Mean', 'Variance'])
    for i in range(1, len(dataset)):
        df_plot.loc[i] = [i, np.mean(dataset[column_name].head(i)), np.var(dataset[column_name].head(i))]
    plt.subplot(2, 1, 1)
    plt.plot(df_plot['Samples'], df_plot['Mean'], label='Rolling Mean')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean - {}'.format(column_name))
    plt.subplot(2, 1, 2)
    plt.plot(df_plot['Samples'], df_plot['Variance'], label='Rolling Variance')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance - {}'.format(column_name))
    plt.tight_layout()
    plt.show()
# The following function calculates the Rolling Mean and Rolling Variance and subsequently plots the graph
def Graph_rolling_meanvar(dataset, column_name):
    df_plot = pd.DataFrame(columns=['Samples', 'Mean', 'Variance'])
    for i in range(1, len(dataset)):
        df_plot.loc[i] = [i, np.mean(dataset[column_name].head(i)), np.var(dataset[column_name].head(i))]
    plt.subplot(2, 1, 1)
    plt.plot(df_plot['Samples'], df_plot['Mean'], label='Rolling Mean')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Mean - {}'.format(column_name))
    plt.subplot(2, 1, 2)
    plt.plot(df_plot['Samples'], df_plot['Variance'], label='Rolling Variance')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variance - {}'.format(column_name))
    plt.tight_layout()
    plt.show()

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


def differencing(series, order=1):
    diff = []
    for i in range(order):
        diff.append(np.nan)
    for i in range(order, len(series)):
        diff.append(series[i] - series[i - 1])
    return diff

def seasonal_differencing(series, seasons):
    diff = []
    for i in range(seasons):
        diff.append(np.nan)
    for i in range(seasons, len(series)):
        diff.append(series[i] - series[i - seasons])
    return diff

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
    plt.figure(figsize=(12, 8))
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

def dlsim_method(mean,var,n,num,den):
    np.random.seed(6313)
    system = (num, den, 1)
    e = np.random.normal(mean, var, n)
    t, y_dlsim = signal.dlsim(system, e)
    print(f" Generated data- {y_dlsim[:5].flatten()}")
    return y_dlsim.reshape(-1), e
def Cal_autocorr(y, lag):
    mean = np.mean(y)
    numerator = 0
    denominator = 0
    for t in range(0, len(y)):
        denominator += (y[t] - mean) ** 2
    for t in range(lag, len(y)):
        numerator += (y[t] - mean)*(y[t-lag] - mean)
    return numerator/denominator
def gpac(ry, show_heatmap='Yes', j_max=7, k_max=7, round_off=3, seed=6313):
    np.random.seed(seed)
    gpac_table = np.zeros((j_max, k_max-1))
    for j in range(0, j_max):
        for k in range(1, k_max):
            phi_num = np.zeros((k, k))
            phi_den = np.zeros((k, k))
            for x in range(0, k):
                for z in range(0, k):
                    phi_num[x][z] = ry[abs(j + 1 - z + x)]
                    phi_den[x][z] = ry[abs(j - z + x)]
            phi_num = np.roll(phi_num, -1, 1)
            det_num = np.linalg.det(phi_num)
            det_den = np.linalg.det(phi_den)
            if det_den != 0 and not np.isnan(det_den):
                phi_j_k = det_num / det_den
            else:
                phi_j_k = np.nan
            gpac_table[j][k - 1] = phi_j_k #np.linalg.det(phi_num) / np.linalg.det(phi_den)
    if show_heatmap=='Yes':
        plt.figure(figsize=(16, 8))
        x_axis_labels = list(range(1, k_max))
        sns.heatmap(gpac_table, annot=True, xticklabels=x_axis_labels, fmt=f'.{round_off}f', vmin=-0.1, vmax=0.1)#, cmap='BrBG'
        plt.title(f'GPAC Table', fontsize=18)
        plt.show()
    #print(gpac_table)
    return gpac_table
def gpac_table(ry,na,nb,j_max=7, k_max=7,lag=15):
    ry1 = ry[::-1]
    ry2 = np.concatenate((ry1[-lag:], ry[:lag+1]))
    gpac_table = np.zeros((j_max, k_max - 1))
    for j in range(0, j_max):
        for k in range(1, k_max):
            phi_num = np.zeros((k, k))
            phi_den = np.zeros((k, k))
            for x in range(0, k):
                for z in range(0, k):
                    phi_num[x, z] = ry2[j - z + x + 1 + lag - 1]
                    phi_den[x, z] = ry2[j - z + x + lag - 1]
            phi_num = np.concatenate((phi_num[:, 1:], phi_num[:, 0:1]), axis=1)
            det_num = np.linalg.det(phi_num)
            det_den = np.linalg.det(phi_den)
            if det_den != 0 and not np.isnan(det_den):
                phi_j_k = det_num / det_den
            else:
                phi_j_k = np.nan
            gpac_table[j, k - 1] = "{:.2f}".format(phi_j_k)
    x_labels = list(range(1, k_max))
    sns.heatmap(gpac_table, annot=True,xticklabels=x_labels ,fmt=".2f", cmap="coolwarm",vmin=-0.1,vmax=0.1)
    plt.title(f"GPAC Table for ARMA{na,nb} Process")
    plt.xlabel("k")
    plt.ylabel("j")
    plt.show()
    print(gpac_table)
    return gpac_table
def gpac(ry, show_heatmap='Yes', j_max=7, k_max=7, round_off=3, seed=6313):
    np.random.seed(seed)
    gpac_table = np.zeros((j_max, k_max-1))
    for j in range(0, j_max):
        for k in range(1, k_max):
            phi_num = np.zeros((k, k))
            phi_den = np.zeros((k, k))
            for x in range(0, k):
                for z in range(0, k):
                    phi_num[x][z] = ry[abs(j + 1 - z + x)]
                    phi_den[x][z] = ry[abs(j - z + x)]
            phi_num = np.roll(phi_num, -1, 1)
            gpac_table[j][k - 1] = np.linalg.det(phi_num) / np.linalg.det(phi_den)
    if show_heatmap=='Yes':
        plt.figure(figsize=(16, 8))
        x_axis_labels = list(range(1, k_max))
        sns.heatmap(gpac_table, annot=True, xticklabels=x_axis_labels, fmt=f'.{round_off}f', vmin=-0.1, vmax=0.1)#, cmap='BrBG'
        plt.title(f'GPAC Table', fontsize=18)
        plt.show()
    #print(gpac_table)
    return gpac_table

from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags,method='ywm')
    fig.tight_layout(pad=3)
    plt.show()
# when pacf me cutoff - AR
# ACF CUTOFF - MA

def analyze_arma_process( lags, j_max=7, k_max=7):
    np.random.seed(6313)
    n = int(input("Enter the number of data samples: "))
    mean = int(input("Enter the mean of white noise: "))
    var = int(input("Enter the variance of the white noise: "))
    na = int(input("Enter AR order: "))
    nb = int(input("Enter MA order:"))
    print("Enter the coefficients of AR :")
    ar_coef = [eval(input(f'AR coefficient {i+1}:'))for i in range(na+1)]
    print("Enter the coefficients of MA:")
    ma_coef = [eval(input(f'MA coefficient {i+1}:'))for i in range(nb+1)]
    ar_params = np.r_[ar_coef]
    ma_params = np.r_[ma_coef]
    arma_process = sm.tsa.ArmaProcess(ar_params, ma_params)
    mean_y = mean * (1 + np.sum(ar_coef)) / (1 + np.sum(ma_coef))
    data = arma_process.generate_sample(n, scale=np.sqrt(var)) + mean_y
    formatted_data = [float("{:.2f}".format(num)) for num in data[:15]]

    print(f"Generate ARMA{na, nb} Process", formatted_data)
    acf_y = arma_process.acf(lags=lags)
    acf_y_list = acf_y.tolist()

    formatted_acf_y = [float("{:.2f}".format(num)) for num in acf_y_list[:15]]
    print("Theoretical ACF for y(t) with lags = {}: {}".format(lags, formatted_acf_y))
    gpac_table(acf_y,na,nb,j_max=7, k_max=7)
    ACF_PACF_Plot(data, lags=20)


#LM STEPS
def generate_input():
    n = int(input("Number of observations: "))
    mean = int(input('Enter the mean of white noise: '))
    var = int(input('Enter the variance of white noise: '))
    na = int(input("Enter AR order: "))
    nb = int(input("Enter MA order: "))
    an = [0] * max(na, nb)
    bn = [0] * max(na, nb)
    for i in range(na):
        an[i] = float(input("Enter the coefficient of a{}: ".format(i + 1)))
    for i in range(nb):
        bn[i] = float(input("Enter the coefficient of b{}: ".format(i + 1)))
    ar_params = np.array(an)
    ma_params = np.array(bn)
    ar = np.r_[1, ar_params]
    ma = np.r_[1, ma_params]
    print('AR coefficient:', ar)
    print('MA coefficient:', ma)
    return n,mean,var,na,nb,ar, ma

def calculate_error(y, na, theta):
    np.random.seed(6313)
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):
        for x in range(len(den) - len(num)):
            num = np.append(num, 0)
    elif len(num) > len(den):
        for x in range(len(num) - len(den)):
            den = np.append(den, 0)
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    return e

def compute_lm_step1(y, na, nb, delta, theta):
    n1 = na + nb
    e = calculate_error(y, na, theta)
    sse_old = np.dot(e.T, e)[0][0]
    X = np.empty((len(y), n1))
    for i in range(n1):
        theta[i] += delta
        e_i = calculate_error(y, na, theta)
        x_i = (e - e_i) / delta
        X[:, i] = x_i[:, 0]
        theta[i] -= delta
    A = np.dot(X.T, X)
    g = np.dot(X.T, e)
    return A, g, X, sse_old

def compute_lm_step2(y, na, A, theta, mu, g):
    delta_theta = np.linalg.solve(A + mu * np.eye(A.shape[0]), g)
    theta_new = theta + delta_theta
    e_new = calculate_error(y, na, theta_new)
    sse_new = np.dot(e_new.T, e_new)[0][0]
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    return sse_new, delta_theta, theta_new

def compute_lm_step3(y, na, nb):
    N = len(y)
    n = na + nb
    mu = 0.01
    mu_max = 10 ** 20
    max_iterations = 100
    delta = 10 ** -6
    var_e = 0
    covariance_theta_hat = 0
    sse_list = []
    theta = np.zeros((n, 1))

    for iterations in range(max_iterations):
        A, g, X, sse_old = compute_lm_step1(y, na, nb, delta, theta)
        sse_new, delta_theta, theta_new = compute_lm_step2(y, na, A, theta, mu, g)
        sse_list.append(sse_old)
        if iterations < max_iterations:
            if sse_new < sse_old:
                if np.linalg.norm(delta_theta, 2) < 10 ** -3:
                    theta_hat = theta_new
                    var_e = sse_new / (N - n)
                    covariance_theta_hat = var_e * np.linalg.inv(A)
                    print("Convergence Occured")
                    break
                else:
                    theta = theta_new
                    mu /= 10
            while sse_new >= sse_old:
                mu = mu * 10
                if mu > mu_max:
                    print('No Convergence')
                    break
                sse_new, delta_theta, theta_new = compute_lm_step2(y, na, A, theta, mu, g)
        else:
            print('Max Iterations Reached')
            break
        theta = theta_new
    return theta_new, sse_new, var_e, covariance_theta_hat, sse_list



def lm_confidence_interval(theta, cov, na, nb):
    print("Confidence Interval for the Estimated Parameters")
    lower_bound = theta - 2 * np.sqrt(np.diag(cov))
    upper_bound = theta + 2 * np.sqrt(np.diag(cov))
    round_off = 3
    lower_bound = np.round(lower_bound, decimals=round_off)
    upper_bound = np.round(upper_bound, decimals=round_off)
    for i in range(na + nb):
        if i < na:
            print(f"AR Coefficient {i+1}: ({lower_bound[i][0]}, {upper_bound[i][0]})")
        else:
            print(f"MA Coefficient {i + 1 - na}: ({lower_bound[i][0]}, {upper_bound[i][0]})")

def find_roots(theta, na):
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):
        num = np.pad(num, (0, len(den) - len(num) - 1), mode='constant')
    elif len(num) > len(den):
        den = np.pad(den, (0, len(num) - len(den) - 1), mode='constant')
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    roots_num = np.round(np.roots(num), decimals=3)
    roots_den = np.round(np.roots(den), decimals=3)
    print("Poles :", roots_num)
    print("Zeros:", roots_den)


def plot_sse(sse_list):
    plt.plot(sse_list)
    plt.xlabel('Iterations')
    plt.ylabel('SSE')
    plt.title('SSE over the iterations')
    plt.show()

def estimate_arma(mean,var,na,nb,ar,ma, n):
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    mean_y = mean * (1 + np.sum(ma)) / (1 + np.sum(ar))
    y = arma_process.generate_sample(n,scale=np.sqrt(var)) + mean_y
    model = sm.tsa.ARIMA(y, order=(na, 0, nb), trend=None).fit()
    for i in range(na):
        print("The AR coefficient a{} is: {:.3f}".format(i, -model.arparams[i]))
    for i in range(nb):
        print("The MA coefficient b{} is: {:.3f}".format(i+1, model.maparams[i]))
    print(model.summary())


delta_num_den = 1e-20
def num_den_size(num, den):
    if len(num) > len(den):
        den = den + [delta_num_den] * (len(num) - len(den))
    elif len(num) < len(den):
        num = num + [delta_num_den] * (len(den) - len(num))
    return num, den