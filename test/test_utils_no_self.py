import os
import unittest

import pandas as pd
import numpy as np

from daa.utils import *

cwd =  r'C:\devl\DAA\test'
# cwd = os.getcwd()
print(cwd)

# User inputs: date range, strategy and User ID
dates = pd.bdate_range(start = '1/4/2016', end ='12/17/2019') #YYYYMMDD
strategy = {'name': 'EqualWeight', 'rebalance': 'Limit',
            'lookback': '1YR', 'limit': 0.05}
ID = 1 # user ID
initial_cash = 1e6

# Create Exchange and Portfolio objects
exchange = Exchange(os.path.join(cwd,'testdata/PriceData.csv'))
portfolio = Portfolio(dates[0], exchange, initial_cash, ID) 
assert portfolio.cash_balance == 1e6, "Cash incorrect!"

backtest = Backtest(exchange, portfolio, strategy, dates, ID)
backtest.run()

# TODO: Create Data Visualization Class
# Print table of portfolio statistics vs. generic benchmarks
# Plot cumulative total return series vs. generics
# Plot rolling max drawdowns
blotter_df = portfolio.get_trade_blotter()
positions = portfolio.get_positions_df()
price_df = exchange.get_price_df()

portfolio.pass_time(6)
portfolio.place_order('SPX_Index', 'buy', 5, 'market', exchange)
portfolio.place_order('EAFE_Index', 'buy', 5, 'market', exchange)
portfolio.place_order('Small_Growth_Index', 'buy', 5, 'market', exchange)
portfolio.place_order('EM_Index', 'buy', 5, 'market', exchange)
portfolio.pass_time(12)
print(portfolio.cash_balance)
positions = portfolio.get_positions_df()        
assert positions.loc['SPX_Index','quantity'] == 5, "Shares incorrect!"
portfolio.place_order('SPX_Index', 'sell', 3, 'market', exchange)
portfolio.pass_time(24) 
positions = portfolio.get_positions_df() 
assert positions.loc['SPX_Index','quantity'] == 2, "Remaining shares incorrect!"

#OTHER TESTS:
positions = portfolio.get_positions_df()   
ds = pd.to_datetime('2016-01-04 00:00:00')

for idx in range(len(positions)):
    ticker = positions.index[idx]
    price = exchange.get_price(ticker, ds)
    positions.loc[positions.index[idx], 'price'] = price
    
positions['value'] = positions['price']*positions['quantity']
positions['wgt'] = positions['value']/positions['value'].sum()
