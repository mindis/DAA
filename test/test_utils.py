import os, sys
import unittest

# Make sure to run utils.py first
from utils import *

cwd = os.getcwd()

class UtilTests(unittest.TestCase):

    def setUp(self):
        print("setting up")
        pass

    def test_my_var(self):
        self.assertEqual(send_four(), 4)

# Assume Market Orders only

class PortfolioTests(unittest.TestCase):
    
    def setUp(self):
        self.exchange = Exchange(os.path.join(cwd,
                                              'testdata/PriceData.csv'))
        self.portfolio = Portfolio('2016-01-04 00:00:00') # Portfolio starts empty 
        # Assuming that Portfolio is not responsible for handling market
        # orders, limit orders, etc. 

    def tearDown(self):
        del self.portfolio

    def test_increment_time(self):
        self.portfolio.pass_time(1)
        self.assertEqual(str(self.portfolio.ds), '2016-01-04 00:00:00')

    def test_add_cash(self):
        self.assertEqual(self.portfolio.cash_balance, 0)
        self.portfolio.add_cash(10000)
        self.assertEqual(self.portfolio.cash_balance, 10000)

    def test_order(self):

        self.portfolio = Portfolio('2016-01-04 00:00:00') # Portfolio starts empty 

        self.portfolio.add_cash(1000)
        self.portfolio.pass_time(6)
        self.portfolio.place_order('SPTR', 'buy', 2, 'market', self.exchange)
        self.portfolio.place_order('AGG', 'buy', 2, 'market', self.exchange)
        #                        slippage=False)
        #                        slippage=False)
        
        self.portfolio.pass_time(12)
        
        self.assertEqual(self.portfolio.cash_balance, 780)
        self.assertEqual(self.portfolio.get_positions_df['SPTR'], 100)

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


#class StrategyTests(unittest.TestCase):
#    
#    def setUp(self):
#        exchange = Exchange(os.path.join(THIS_DIR,
#                                         'testdata/PriceData.csv'))
#        self.portfolio = Portfolio('2001-08-01 00:00:00') # Portfolio starts empty 
#        self.portfolio.add_cash(1000)
#
#        # Does a Strategy need a portfolio, or does a portfolio need a strategy?
#
#        self.strategy = Strategy(exchange)
#
#
#    def test_buy_and_hold_single_equity(self):
#        buy_and_hold = self.strategy.buy_and_hold('SPTR', slippage=False)
#        self.portfolio.implement_strategy(buy_and_hold)
#        self.portfolio.pass_time(66)
#
#        self.assertEqual(self.portfolio.equity_balance['SPTR'], 1200)
#
#    def test_buy_and_hold_multiple_equities(self):
#        buy_and_hold = self.strategy.buy_and_hold(['SPTR', 'AGG'], [.55, .4])
#                                            #     slippage=False, interest = 0)
#
#        self.portfolio.implement_strategy(buy_and_hold)
#        self.strategy.portfolio.pass_time(66)
#
#        self.assertEqual(self.portfolio.equity_balance['SPTR'], 660)
#        self.assertEqual(self.portfolio.cash_balance, 50)


