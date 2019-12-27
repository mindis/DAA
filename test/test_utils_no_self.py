import os
import unittest

from daa.utils import *

cwd =  r'C:\Users\halls\DAA\test'
#cwd = os.getcwd()
print(cwd)

# Assume market on close orders only

exchange = Exchange(os.path.join(cwd,'testdata/PriceData.csv'))
portfolio = Portfolio('2016-01-04 00:00:00') # Portfolio starts empty 
price_df = exchange.get_price_df()
portfolio.add_cash(100000)
portfolio.pass_time(6)
portfolio.place_order('SPX_Index', 'buy', 5, 'market', exchange)
portfolio.place_order('EAFE_Index', 'buy', 5, 'market', exchange)
portfolio.place_order('Small_Growth_Index', 'buy', 5, 'market', exchange)
portfolio.place_order('EM_Index', 'buy', 5, 'market', exchange)
portfolio.pass_time(12)
print(portfolio.cash_balance)
positions = portfolio.get_positions_df()        
assert positions['SPX_Index'] == 5, "Number of shares incorrect!"
portfolio.place_order('SPX_Index', 'sell', 3, 'market', exchange)
portfolio.pass_time(24) 
positions = portfolio.get_positions_df() 
assert positions['SPX_Index'] == 2, "Remaining shares incorrect!"

