import os
import unittest


from daa.strategy import *
from daa.utils import *

class StrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'testdata/PriceData.csv'))

        portfolio = Portfolio('2019-01-01', exchange, 100000, 0)
        portfolio.place_order('SPX_Index', 'buy', 5, 'market', exchange)
        portfolio.pass_time()

        strategy = Strategy1(exchange)
        actual_weights = strategy.compute_actual_weights(portfolio)
        target_weights = strategy.compute_target_weights(portfolio)
        
