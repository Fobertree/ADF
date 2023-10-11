import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import threading
import concurrent.futures
import statsmodels.api as sm

lock = threading.Lock()

dest_coint = "ADF_Cointegrated"
dest_not_coint = "ADF_Diff"
csv_path = "spy.csv"
max_threads = 10

spy = pd.read_csv(csv_path)
spy.dropna(how="all",axis=1)

tickers = [i for i in spy.columns]
tickers.pop(0) #pop column name for index

print(tickers)

if not os.path.exists(dest_coint):
    os.mkdir(dest_coint)

def regress_two(t1: str,t2: str): # regress t1 on t2 to get cointegration vector
    Y = spy[t1]
    X = spy[t2]

    Y.to_list()
    X.to_list()

    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()

    print(results.params, results.params.to_list()[0][1])

    return results.params.to_list()[0][1]


#if not os.path.exists(dest_not_coint):
#    os.mkdir(dest_not_coint)

# Dickey Fuller and ADF can only be done on a snigle time series. It is referred to as a test of stationarity for this reason
# If we merge two time series, then if the test for stationary rejects null hypothesis (residual is stationary), then cointegration is proven
def compare_two(t1: str,t2: str):
    with lock:
        df1 = spy[t1] #returns series, not dataframe
        df2 = spy[t2]
        try:
            combined = pd.merge(df1,df2,how="inner",left_index=True,right_index=True)

            beta = regress_two(t1,t2) # regress t1 on t2

            combined = combined[t1] - beta * combined[t2] #get residuals

            # Stationarity: Yt - \beta * Xt = I(0)

            plt.plot(combined)
            result = adfuller(combined)

            #print(result)
            #input()

            print(f'ADF Statistic: {result[0]:.2f}')
            print(f'p-value: {result[1]:.2f}')
            print('Critical Values:')
            for key,value in result[4].items():
                print(f'\t{key}: {value:.2f}')
            
            print(f'Tickers {t1}, {t2}')

            filename = f"ADF_{t1}_{t2}.png"

            if result[0] < result[4]["5%"]:
                print("Reject H0 - Time Series is Stationary")
                plt.savefig(os.path.join(dest_coint,filename))
            else:
                print("Failed to Reject H0 - Time Series is Non-Stationary")
                #plt.savefig(os.path.join(dest_not_coint,filename))

            plt.clf()
            '''plt.show()
            time.sleep(5)
            plt.close()'''

        except Exception as e:
            print(e)

with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
    for i in tqdm(range(len(tickers))):
        for j in range(i+1,len(tickers)):
            executor.submit(compare_two,tickers[i],tickers[j])
            #thread = threading.Thread(target=compare_two, args=(tickers[i],tickers[j]))
            #thread.start()
            #compare_two(tickers[i],tickers[j])
