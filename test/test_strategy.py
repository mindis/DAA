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
        strategy = MomentumStrategy(exchange, schedule='D',
                                    ticker_dict={'SPY': 1, 'AGG': 0},
                                    lookback=200)

        backtest = Backtest(exchange, strategy, '1996-01-04', '2020-04-15')
        backtest.run()

        plt.style.use('fivethirtyeight')

        val_df = backtest.value_df.copy().reset_index()
        ex_df = backtest.exchange.price_df.copy().reset_index()
        ex_df['Cash'] = 1.0  # TODO: should we have inflation baked in?        

        tickers = list(set(val_df.ticker))
        for ticker in tickers:
            ticker_df = val_df.loc[val_df.ticker == ticker] 

            fig, ax = plt.subplots()
            ax.set_title(f'Portfolio Profile: {ticker}')
            ax.set_ylim(0, 4)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_ylabel(f'Weight of {ticker} in portfolio')
            ax.step(ticker_df['Date'], ticker_df['wgt'], color='green',
                    linewidth=.8)
            ax2 = ax.twinx()
            ax2.grid(b=False)
            ax2.set_ylabel(f'Total Returns Price of {ticker}')
            ax2.plot(ex_df['Date'], ex_df[ticker], linewidth=.8)

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
