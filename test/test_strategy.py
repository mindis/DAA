import os
import unittest

from daa.strategy import *
from daa.utils import *
from daa.portfolio import *
from daa.exchange import *
from daa.backtest import *

class FixedWeightStrategy(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'../data/price_data_yf.csv'))
        strategy = Fixed_Weights(exchange, 'M', 
                                 ticker_dict={'SPY': 0.70, 'AGG': 0.30})
        backtest = Backtest(exchange, strategy, '2005-01-02', '2020-04-15')
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
        strategy = MomentumStrategy(exchange, schedule='M',
                                    ticker_dict={'SPY': 1, 'AGG': 0},
                                    lookback=200)

        backtest = Backtest(exchange, strategy, '2005-01-02', '2020-04-15')
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
        exchange = Exchange(os.path.join(cwd,'../data/price_data_ef.csv'))
        strategy = Max_Sharpe(exchange, 'M',
         ['VLUE', 'QUAL', 'SIZE', 'USMV', 'MTUM'], 252)
        backtest = Backtest(exchange, strategy, '2015-01-02', '2020-04-15')
        backtest.run()

if __name__ == '__main__':
    unittest.main()
