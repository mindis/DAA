import os
import unittest

from daa.strategy import Strategy1, Strategy2, Strategy3
from daa.utils import *
from daa.portfolio import *
from daa.exchange import *
from daa.backtest import *

class StrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))
        start_dt = '1999-01-04'
        end_dt = '2005-12-31'
        # Day 1
        portfolio = Portfolio(start_dt, exchange, 100000, 0)
        strategy = Strategy1(exchange, 'M')

        backtest = Backtest(portfolio, exchange, strategy)
        backtest.run(end_dt)

        portfolio.get_positions_df()
           
class Strategy2Test(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        start_dt = '1999-01-04'
        end_dt = '2001-12-31'
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))
        portfolio = Portfolio(start_dt, exchange, 1000000, 0)
        strategy = Strategy2(exchange, 'M')
        backtest = Backtest(portfolio, exchange, strategy)
        backtest.run(end_dt)
        
        portfolio.get_trade_blotter()
        portfolio.get_positions_df()
        strategy.get_trcape(portfolio)

# TODO(baogorek): 1. Fix data source issue, 2. make tests runnable with pytest

class Strategy3Test(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        start_dt = '1996-01-04'
        end_dt = '2001-10-30'
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))
        portfolio = Portfolio(start_dt, exchange, 1000000, 0)

        strategy = Strategy3(exchange, 'M',
                {'SPX_Index': 1, 'Tsy_Index': 0, 'EM_Index': 1, 'MBS_Index': 0}, 200)

        backtest = Backtest(portfolio, exchange, strategy)
        backtest.run(end_dt)
        
        portfolio.get_trade_blotter()
        portfolio.get_positions_df()


def trader(actual_weights, target_weights, price_dict, value):

     if 'Cash'actual_weight
     tickers = set(actual_weights.keys()) - {'Cash'}
     price_dict['Cash'] = 1.

     delta_weights = {}
     shares_to_trade = {}
     for ticker in tickers:
         delta_weights[ticker] = target_weights[ticker] - actual_weights[ticker]
         shares = value * delta_weights[ticker] / price_dict[ticker]
         shares = np.sign(shares) * np.floor(np.abs(shares))
         if np.abs(shares) > 0:
             shares_to_trade[ticker] = shares

     return shares_to_trade         


trader({'Cash':0., 'A':.5,'B': .5}, {'A': 1, 'B': 0}, {'A': 10, 'B':10}, 1000)

if __name__ == '__main__':
    unittest.main()
