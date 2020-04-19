import numpy as np
import pandas as pd

import yfinance as yf

data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = "SPY VLUE QUAL SIZE USMV MTUM",

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "10y",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1d",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

# Clean data
df_clean = data[[('SPY', 'Close'), ('VLUE', 'Close'), ('QUAL', 'Close'), 
                ('SIZE', 'Close'), ('USMV', 'Close'), ('MTUM', 'Close')]]
df_clean.reset_index(inplace=True)
df_clean.columns=['Date', 'SPY', 'VLUE', 'QUAL', 'SIZE', 'USMV', 'MTUM']
df_clean = df_clean.dropna()

df_clean.to_csv('price_data_equity_factors.csv', index=False)

