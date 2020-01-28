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

        #TODO (baogorek): How can we simulate one order without pass_time?

        #portfolio.place_order('SPX_Index', 'buy', 5, 'market', exchange)
        #portfolio.place_order('EAFE_Index', 'buy', 5, 'market', exchange)
        #portfolio.place_order('EM_Index', 'buy', 5, 'market', exchange)
        #portfolio.place_order('Large_Value_Index', 'buy', 5, 'market', exchange)
        #portfolio.pass_time()

        #strategy = Strategy1(exchange, 'M')
        #actual_weights = strategy.compute_actual_weights(portfolio)
        #target_weights = strategy.compute_target_weights(portfolio)
        # Day 1 trades
        #trades = strategy.calculate_trades(portfolio)
        #for ticker in trades.keys():
        #    side = 'buy' if trades[ticker] > 0 else 'sell'
        #    portfolio.place_order(ticker, side, np.abs(trades[ticker]),
        #                          'market', exchange)
        #portfolio.pass_time()
        #portfolio.get_positions_df()

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
