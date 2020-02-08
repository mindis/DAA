import os
import unittest

from daa.strategy import Strategy1, Strategy2
from daa.utils import *
from daa.portfolio import *
from daa.exchange import *
from daa.backtest import *

class StrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'testdata/PriceData.csv'))

        # Day 1
        portfolio = Portfolio('2019-01-01', exchange, 100000, 0)
        strategy = Strategy1(exchange, 'M')

        backtest = Backtest(portfolio, exchange, strategy)
        backtest.run('2019-09-02')

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
        start_dt = '1999-01-04'
        end_dt = '2001-12-31'
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))
        portfolio = Portfolio(start_dt, exchange, 1000000, 0)


        strategy = Strategy3(exchange, 'M', {'SPX_Index': 1, 'Tsy_Index': 0, 'EM_Index': 1}, 200)

        backtest = Backtest(portfolio, exchange, strategy)
        backtest.run(end_dt)
        
        portfolio.get_trade_blotter()
        portfolio.get_positions_df()
        strategy.get_trcape(portfolio)


if __name__ == '__main__':
    unittest.main()
