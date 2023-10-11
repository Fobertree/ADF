import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
from requests import Session
from requests_cache import CacheMixin,SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import time

start_time = time.time()

class CachedLimiterSession(CacheMixin,LimiterMixin,Session): #inherits three classes
    pass

session = CachedLimiterSession(
    limiter= Limiter(RequestRate(2,Duration.SECOND*5)), #max 2 requests per 5 seconds. Yahoo Finance API seems to rate limit to 2000 requests per hour
    bucket_class = MemoryQueueBucket,
    backend= SQLiteCache('yfinance.cache')
)

tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
print(tickers.head())

spy_data = yf.download(tickers.Symbol.to_list(),'2021-1-1','2021-7-12',auto_adjust=True,session=session)['Close']
print(spy_data.head())

spy_data.to_csv('spy.csv',index=True)


print(f'Took {time.time()-start_time} seconds.')