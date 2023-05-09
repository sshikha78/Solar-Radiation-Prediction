import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from scipy import signal
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error


def plot_graph(x_values, y_values, x_label='', y_label='', title='', legend_label='', figsize=(16, 8)):
    plt.figure(figsize=figsize)
    plt.plot(x_values, y_values, label=legend_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    fig, ax = plt.subplots(2, 1,figsize=(12, 8))
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
    plt.figure(figsize=(12, 8))
    markers, stemlines, baseline = plt.stem(lag_f, final)
    plt.setp(markers, color='red', marker='o')
    plt.axhspan((-1.96 / np.sqrt(len(y))), (1.96 / np.sqrt(len(y))), alpha=0.2, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(f'AutoCorrelation of {method_name}')
    plt.show()

def ACF_PACF_Plot(y, lags, method_name=''):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title(f'ACF/PACF of the {method_name} data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


yt_pred = []
error = []
e_squared=[]

def seasonal_differencing(series, seasons=1):
    diff = []
    for i in range(seasons):
        diff.append(np.nan)
    for i in range(seasons, len(series)):
        diff.append(series[i] - series[i - seasons])
    return diff


def average(t,yt,n):
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
        #if i == 0:
        if pred[i] == np.nan:
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

    print(f'Mean of residual error is {np.round(res_mean, 2)}')
    print(f'MSE of residual error for  is {np.round(mse_tr, 2)}')
    print(f'Variance of residual error   is {np.round(var_pred, 2)}')
    print(f'Variance of forecast error  is {np.round(var_fcst, 2)}')
    print(f'Ratio of variance of residual errors versus variance of forecast errors : {np.round(var_pred / var_fcst, 2)}')

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

def naive(yt, n,h):
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
    return pd.Series(yt_pred_d)

# drift method
def drift(y, n):
    y_pred = list(np.nan for i in range(0, len(y)))
    for i in range(2, n):
        y_pred[i] = y[i-1] + ((y[i-1]-y[0]))/(i-1)
    for i in range(n, len(y)):
        y_pred[i] = y[n-1] + (i+1-n)*(y[n-1]-y[0])/(n-1)
    return y_pred

def ses(t,yt,n,alpha):
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



def gpac_table(n, mean, var, ar_coef, ma_coef, j_max=7, k_max=7):
    ar_params = np.r_[ar_coef]
    ma_params = np.r_[ma_coef]
    arma_process = sm.tsa.ArmaProcess(ar_coef, ma_coef)
    mean_y = mean * (1 + np.sum(ar_coef)) / (1 + np.sum(ma_coef))
    y = arma_process.generate_sample(n, scale=np.sqrt(var)) + mean_y
    lags = 15
    ry = arma_process.acf(lags=lags)
    ry1 = ry[::-1]
    ry2 = np.concatenate((np.reshape(ry1, lags), ry[1:]))
    gpac_table = np.zeros((j_max, k_max - 1))
    for j in range(0, j_max):
        for k in range(1, k_max):
            phi_num = np.zeros((k, k))
            phi_den = np.zeros((k, k))
            for x in range(0, k):
                for z in range(0, k):
                    phi_num[x, z] = ry2[j - z + x + 1 + lags - 1]
                    phi_den[x, z] = ry2[j - z + x + lags - 1]
            phi_num = np.roll(phi_num, -1, 1)
            phi_j_k = round(np.linalg.det(phi_num) / np.linalg.det(phi_den), 3)
            gpac_table[j, k - 1] = phi_j_k
    sns.heatmap(gpac_table, annot=True, fmt=".3f", cmap="coolwarm")
    plt.xlabel("k")
    plt.ylabel("j")
    plt.show()
    return gpac_table



from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags,method_name):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title(f'ACF/PACF of the {method_name}')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags,method='ywm')
    fig.tight_layout(pad=3)
    plt.show()


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



def stl_decomposition(df, col, period=365, plot=True):
    stl = STL(df, period=period)
    res = stl.fit()
    T, S, R = res.trend, res.seasonal, res.resid
    if plot:
        plt.figure(figsize=(12, 8))
        fig = res.plot()
        plt.xlabel('Year')
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(df.index, T.values, label='trend')
        plt.plot(df.index, S.values, label='Seasonal')
        plt.plot(df.index, R.values, label='residuals')
        plt.xlabel("Year")
        plt.ylabel(col)
        plt.legend()
        plt.grid()
        plt.show()

        adj_seasonal = (df.values - res.seasonal)
        plt.figure(figsize=(12, 8))
        plt.plot(df.values, label='Original Data')
        plt.plot(adj_seasonal.values, label='Adjusted Seasonally')
        plt.legend()
        plt.title('Original Data vs Seasonally Adjusted Data')
        plt.xlabel("Year")
        plt.ylabel(col)
        plt.grid()
        plt.show()

        detrended = (df.values - res.trend)
        plt.figure(figsize=(12, 8))
        plt.plot(df, label='Original Data')
        plt.plot(detrended, label='Detrended Data')
        plt.title(f'Detrended Data vs Original Data of {col}')
        plt.xlabel('Frequency')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()

    var_resi1 = np.var(R)
    var_resid_trend = np.var(T + R)
    Ft = np.max([0, 1 - var_resi1 / var_resid_trend])
    print(f"The strength of trend for {col} is {Ft:.2f}")

    var_resi = np.var(R)
    var_resid_seasonal = np.var(S + R)
    St = np.max([0, 1 - var_resi / var_resid_seasonal])
    print(f"The strength of seasonality for {col} is {St:.2f}")


import statsmodels.tsa.holtwinters as ets
def holt_winters_forecast(train,test):
    holtt1 = ets.ExponentialSmoothing(train,).fit()
    pred_train_holts = holtt1.predict(start=0, end=(len(train) - 1))
    pred_test_holts = holtt1.forecast(steps=len(test))
    plt.figure()
    plt.plot(train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in train] + [x for x in test], label='test set')
    plt.plot([None for i in train] + [x for x in pred_test_holts], label='h-step forecast')
    plt.legend()
    plt.title('Temperature - Holt-Winter Seasonal Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()


def forecast_method(arr_train, arr_test, model, alpha=0.5, n=10, lags=20):
    arr_train = np.array(arr_train)
    arr_test = np.array(arr_test)

    if model == 'Average':
        y_pred = []
        for i in range(len(arr_train) + len(arr_test)):
            if i == 0:
                y_pred.append(np.nan)
            elif i <= n:
                y_pred.append(sum(arr_train[:i]) / i)
            else:
                y_pred.append(sum(arr_train[i - n:i]) / n)

        e, e2, mse_tr, mse_ts, var_pred, var_fcst, res_mean = error_method(arr_train, y_pred[:len(arr_train)], n, 1)

    elif model == 'Naive':
        y_pred = naive(arr_train, n, len(arr_test))
        e, e2, mse_tr, mse_ts, var_pred, var_fcst, res_mean = error_method(arr_train, y_pred[:len(arr_train)], n, 1)

    elif model == 'Drift':
        y_pred = drift(arr_train, n) + [np.nan] * len(arr_test)
        e, e2, mse_tr, mse_ts, var_pred, var_fcst, res_mean = error_method(arr_train, y_pred[:len(arr_train)], n, 2)

    elif model == 'SES':
        y_pred = ses(arr_train, arr_train, n, alpha) + [np.nan] * len(arr_test)
        e, e2, mse_tr, mse_ts, var_pred, var_fcst, res_mean = error_method(arr_train, y_pred[:len(arr_train)], n, 1)

    else:
        print(f"Invalid model choice: {model}")
        return None
    Q = sm.stats.acorr_ljungbox(e[2:len(arr_train)], lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print(f"Q-Value for training set ({model} Method) : ", np.round(Q, 2))
    plt.figure()
    plt.plot(arr_train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in arr_train] + [x for x in arr_test], label='test set')
    plt.plot([None for i in arr_train] + y_pred, label='h-step forecast')
    plt.legend()
    plt.title(f'Temperature - {model.capitalize()} Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()


    return y_pred, e, e2, mse_tr, mse_ts, var_pred, var_fcst, res_mean


def Gpac(ry, j_max=7, k_max=7):
    np.random.seed(6313)
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
            gpac_table[j][k - 1] = phi_j_k
    plt.figure(figsize=(36, 28))
    x_axis_labels = list(range(1, k_max))
    sns.heatmap(gpac_table, annot=True, xticklabels=x_axis_labels, fmt=f'.{3}f', vmin=-0.1, vmax=0.1)#, cmap='BrBG'
    plt.title(f'GPAC Table', fontsize=18)
    plt.savefig("gpac.png")
    plt.show()
    return gpac_table

def check_residuals(Q, lags, na, nb):
    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)

    if Q < chi_critical:
        print("As Q-value is less than chi-2 critical, Residual is white")
    else:
        print("As Q-value is greater than chi-2 critical, Residual is NOT white")


def print_coefficients_and_intervals(model, na, nb):
    for i in range(na):
        print(f'The AR coefficient a{i} is: {-1 * model.params[i]}')
    for i in range(nb):
        print(f'The MA coefficient a{i} is: {model.params[i + na]}')

    for i in range(1, na + 1):
        print(f"The confidence interval for a{i} is: {-model.conf_int()[i][0]} and {-model.conf_int()[i][1]}")

    for i in range(1, nb + 1):
        print(f"The confidence interval for b{i} is: {model.conf_int()[i + na][0]} and {model.conf_int()[i + na][1]}")

def plot_train_and_fitted_data(y_train, model_hat, na, d, nb):
    plt.figure()
    plt.plot(y_train, 'r', label='Train data')
    plt.plot(model_hat, 'b', label='Fitted data')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title(f'Stats ARIMA ({na},{d},{nb}) model and predictions')
    plt.grid()
    plt.show()


def plot_test_and_forecast(y_test, test_forecast, na, d, nb):
    plt.figure()
    plt.plot(y_test, 'r', label='Test data')
    plt.plot(test_forecast, 'b', label='Forecasted data')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title(f'Stats ARIMA ({na},{d},{nb}) model and Forecast')
    plt.grid()
    plt.show()




def ARIMA_method(na, nb, d, lags, y_train, y_test):
    model = ARIMA(y_train, order=(na, 0, 0), seasonal_order=(0,0,nb,24)).fit()
    print(model.summary())
    model_hat = model.predict()
    test_forecast = model.forecast(len(y_test))
    error_method(pd.concat([y_train, y_test]), pd.concat([model_hat, test_forecast]), len(y_train), 0)
    ry = sm.tsa.stattools.acf(model.resid, nlags=lags)
    Auto_corr_plot(ry,50,"Arima")
    Q = sm.stats.acorr_ljungbox(model.resid, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for ARIMA residuals: ", Q)
    check_residuals(Q, lags, na, nb)
    print_coefficients_and_intervals(model, na, nb)
    plot_train_and_fitted_data(y_train, model_hat, na, d, nb)
    plot_test_and_forecast(y_test, test_forecast, na, d, nb)
    lbvalue, pvalue = sm.stats.acorr_ljungbox(model.resid, lags=[lags])
    print(f'lbvalue={lbvalue}')
    print(f'pvalue={pvalue}')


def calc_GPAC(acfs, J=10, K=10, plot=True, title=None, savepath=None):
    acfs = np.concatenate((acfs[::-1], acfs[1:]))
    c = acfs.shape[0] // 2
    gpac = np.zeros((J + 1, K))

    for j in range(0, J + 1):
        for k in range(1, K + 1):
            den = np.zeros((k, k))
            for row_index in range(k):
                den[row_index] = acfs[c - j - row_index: c - j + k - row_index]
            vec = acfs[c + j + 1: c + j + k + 1]
            num = den.copy().T
            num[-1] = vec
            num = num.T
            gpac[j, k - 1] = np.divide(np.linalg.det(num), np.linalg.det(den))
    if plot:

        if abs(np.nanmin(gpac)) <= abs(np.nanmax(gpac)):
            vmax = abs(np.nanmax(gpac))
            vmin = -vmax
        else:
            vmax = abs(np.nanmin(gpac))
            vmin = -vmax
        plt.figure(figsize=(60,60))
        sns.heatmap(gpac,
                    annot=True,
                    vmin=vmin, vmax=vmax,
                    cmap='vlag',
                    xticklabels=np.arange(1, K + 1),
                    linewidths=0.5)
        if title:
            plt.title(f'GPAC Table - {title}')
        else:
            plt.title(f'GPAC Table')
        if savepath:
            plt.savefig(savepath)
        plt.show()
    return gpac


def lm(y, na, nb):
    theta, sse, var_error, cov_theta_hat, sse_list = compute_lm_step3(y, na, nb)
    print(theta)

    theta2 = np.array(theta).reshape(-1)
    for i in range(na + nb):
        if i < na:
            ar_coef = "{:.3f}".format(theta2[i])
            print(f"The AR coefficient {i + 1} is: {ar_coef}")
        else:
            ma_coef = "{:.3f}".format(theta2[i])
            print(f"The MA coefficient {i - na + 1} is: {ma_coef}")

    lm_confidence_interval(theta, cov_theta_hat, na, nb)
    cov_theta_hat_rounded = np.round(cov_theta_hat, decimals=3)
    print("Estimated Covariance Matrix of estimated parameters:")
    print(cov_theta_hat_rounded)
    var_error_rounded = round(var_error, 3)
    print("Estimated variance of error:", var_error_rounded)
    find_roots(theta, na)
    plot_sse(sse_list)
