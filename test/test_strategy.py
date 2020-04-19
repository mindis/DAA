import os
import unittest

from daa.strategy import *
from daa.utils import *
from daa.portfolio import *
from daa.exchange import *
from daa.backtest import *


class StrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'../data/price_data_yf.csv'))
        strategy = BasicStrategy(exchange, 'M')
        backtest = Backtest(exchange, strategy, '1999-04-04', '20-12-31')
        backtest.run()


class CAPEStrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))
        strategy = CAPEStrategy(exchange, 'M')
        backtest = Backtest(exchange, strategy, '1999-01-04', '2001-09-03')
        backtest.run()
        

class MomentumStrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'../data/price_data_yf.csv'))
        strategy = MomentumStrategy(exchange, 'D',
            {'SPY': 1, 'EEM': 1, 'AGG': 0}, 100)
        backtest = Backtest(exchange, strategy, '2006-01-04', '2020-04-15')
        backtest.run()

class MinimumVarianceStrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))
        strategy = MinimumVarianceStrategy(exchange, 'M',
         ['SPX_Index', 'EM_Index', 'Large_Value_Index'])

        backtest = Backtest(exchange, strategy, '1998-01-04', '2011-10-30')
        backtest.run()

class MaxSharpeStrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))

        strategy = MaxSharpeStrategy(exchange, 'M',
         ['SPX_Index', 'EM_Index', 'Large_Value_Index', 'AGG_Index'], 252)

        backtest = Backtest(exchange, strategy, '1998-01-04', '2018-10-30')
        backtest.run()

if __name__ == '__main__':
    unittest.main()
