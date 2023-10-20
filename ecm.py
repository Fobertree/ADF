# Autoregressive distributed lag model

# Y_t=a_0+a_1 Y_(t−1)+γ_0 X_t+γ_1 X_(t−1)+u_t

import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

spy = pd.read_csv("spy.csv")
predictors = ["X","const",'disequlibrium']
gammas = set()
alphas = set()

def get_data(t1,t2,m,n):
    data = pd.DataFrame()

    series_X = spy[t1]
    series_X.name = "X"

    close = spy[t2] #series_Y
    close.name = "close"

    data[series_X.name] = series_X

    for lag_order in range(1,m+1):
        new_col= 'gamma_' + str(lag_order)
        data[new_col] = close.shift(lag_order)
        predictors.append(new_col)
        gammas.add(new_col)
    
    for lag_order in range(1,n+1):
        new_col= 'a_' + str(lag_order)
        data[new_col] = close.shift(lag_order)
        predictors.append(new_col)
        alphas.add(new_col)

    #print(data)
    

    data[close.name] = close
    data = data.dropna()

    data_train = data[-50:].reset_index(drop=True)

    data['disequlibrium'] = get_disequilibrium(data_train)

    # diff everything
    data = data.diff().dropna().reset_index(drop=True)
    close = close.diff().dropna().reset_index(drop=True)

    data_train = data[-50:].reset_index(drop=True)

    data_test = data[~data.isin(data_train).all(1)].reset_index(drop=True) #all(1) for columns

    tag = f"Tickers: {t1} {t2}"

    #close = close.reset_index(drop=True)

    print(data,close)
    input()

    return data_train,data_test,tag

def get_disequilibrium(data_train):
    data_train = sm.add_constant(data_train)
    # linear regression on closing share price on the lagged closing market level
    # this is for short-term correction term e^_t-1
    print(data_train)
    lr_model = sm.OLS(data_train['close'], data_train[['const','X']]) # X_(t-1)
    lr_model_fit = lr_model.fit(cov_type='HC0')

    print(lr_model_fit.summary())
    
    print(lr_model_fit.params.keys())
    #const (beta_0), X (beta_1)
    return lr_model_fit.resid #error term

def get_error_correction_coefficient(params): #pi
    # 1-sum(i for i in a[i])
    err_cor_coeff = 1 - sum(params[i] for i in alphas)
    return -err_cor_coeff

def create_ecm():

    data_train,data_test, tag = get_data(spy.columns[1],spy.columns[2],3,3)

    #run training data

    data_train = sm.add_constant(data_train)

    data_train = data_train.reset_index(drop=True)
    
    model = sm.OLS(data_train["close"],data_train[predictors]).fit()

    print(model.summary())

    err_cor_coeff = get_error_correction_coefficient(model.param())


    # add all of terms for sigma by using list comprehension + append
    #fitted model.predict()

def plot_ecm(*func):
    def wrapper():
        pass

    return wrapper

if __name__ == "__main__":
    create_ecm()