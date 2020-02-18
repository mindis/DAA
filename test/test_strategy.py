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
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))
        strategy = BasicStrategy(exchange, 'M')

        backtest = Backtest(exchange, strategy, '1999-04-04', '2005-12-31')
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
        exchange = Exchange(os.path.join(cwd,'../data/PriceData.csv'))

        strategy = MomentumStrategy(exchange, 'D',
            {'SPX_Index': 1, 'Tsy_Index': 0, 'EM_Index': 1, 'MBS_Index': 0},
            200)

        backtest = Backtest(exchange, strategy, '1996-01-04', '2001-10-30')
        backtest.run()


if __name__ == '__main__':
    unittest.main()
