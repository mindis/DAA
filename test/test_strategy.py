import os
import unittest


from daa.strategy import *
from daa.utils import *

class StrategyTest(unittest.TestCase):

    def setUp(self):
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'testdata/PriceData.csv'))

        # Day 1
        portfolio = Portfolio('2019-01-01', exchange, 100000, 0)
        portfolio.place_order('SPX_Index', 'buy', 5, 'market', exchange)
        portfolio.place_order('EAFE_Index', 'buy', 5, 'market', exchange)
        portfolio.place_order('EM_Index', 'buy', 5, 'market', exchange)
        portfolio.place_order('Large_Value_Index', 'buy', 5, 'market', exchange)
        portfolio.pass_time()

        strategy = Strategy1(exchange, 'M')
        actual_weights = strategy.compute_actual_weights(portfolio)
        target_weights = strategy.compute_target_weights(portfolio)
        # Day 1 trades
        trades = strategy.calculate_trades(portfolio)
        for ticker in trades.keys():
            side = 'buy' if trades[ticker] > 0 else 'sell'
            portfolio.place_order(ticker, side, np.abs(trades[ticker]),
                                  'market', exchange)
        portfolio.pass_time()
        portfolio.get_positions_df()

        # Hack backtest using strategy
        cwd = 'test'
        exchange = Exchange(os.path.join(cwd,'testdata/PriceData.csv'))
        strategy = Strategy1(exchange, 'M')
        start_dt = '2016-01-06'
        end_dt = '2017-02-02'
        #portfolio = Portfolio(start_dt, exchange, 100000, 0)
        # TODO: Bought too much EM, running out of cash to fund another buy
        while portfolio.ds < pd.to_datetime(end_dt):
            portfolio.pass_time()
            print(portfolio.ds)
            if exchange.is_end_of_month(portfolio.ds):
                actual_weights = strategy.compute_actual_weights(portfolio)
                target_weights = strategy.compute_target_weights(portfolio)
                trades = strategy.calculate_trades(portfolio)
                for ticker in trades.keys():
                    side = 'buy' if trades[ticker] > 0 else 'sell'
                    portfolio.place_order(ticker, side, np.abs(trades[ticker]),
                                  'market', exchange)    
            
        
        
                
