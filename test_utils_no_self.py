import os
import unittest

# Make sure to run utils.py first
from utils import *

cwd = os.getcwd()

# Assume Market Orders only

exchange = Exchange(os.path.join(cwd,'testdata/PriceData.csv'))
portfolio = Portfolio('2016-01-04 00:00:00') # Portfolio starts empty 

price_df = exchange.get_price_df()

portfolio.add_cash(100000)
portfolio.pass_time(6)
portfolio.place_order('SPX_Index', 'buy', 5, 'market', exchange)
portfolio.place_order('EM_Index', 'buy', 5, 'market', exchange)
portfolio.pass_time(12)
print(portfolio.cash_balance)
positions = portfolio.get_positions_df()        
assertEqual(self.portfolio.cash_balance, 780)
assertEqual(self.portfolio.get_positions_df['SPTR'], 100)

        self.portfolio.pass_time(24) # Should be 6PM 08/02
        self.assertEqual(self.portfolio.equity_balance['SPTR'], 110)

        self.portfolio.place_order('SPTR', 'sell', 1, 'market', self.exchange)
        #  slippage=False)
        self.portfolio.pass_time(24) # Should be 6PM 08/03
        # NOTE: This is where interpreting the single price gets tricky
        # Sold the morning of 8/03, presumably at 8/02's value

        self.assertEqual(self.portfolio.cash_balance, 960)

        # 5 shares that remain are at 08/03's close price
        self.assertEqual(self.portfolio.equity_balance['SPTR'], 60)

