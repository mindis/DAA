import pandas as pd
import numpy as np
from math import sqrt

from tqdm import tqdm
import pdb

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt

class Backtest:
    """Facilitates backtesting strategy on porfolio given exchange"""
    def __init__(self, portfolio, exchange, strategy):
        
        self.portfolio = portfolio
        self.exchange = exchange
        self.strategy = strategy
        self.value_df = pd.DataFrame()
        self.update_value_df() 
           
    def update_value_df(self):
        update = self.portfolio.get_positions_df().reset_index()
        update['Date'] = self.portfolio.ds
        update.set_index('Date', inplace=True)
        self.value_df = pd.concat([self.value_df, update], sort=False)

    def plot_total_return(self):
        plt.plot(self.value_df.groupby('Date').sum()['value'])
        plt.title('Cumulative Growth of Initial Investment')
        plt.show()
        
    def get_value_df(self):
        
        return self.value_df
        
    # def calc_performance(self, freq):
    #     """ Calculate performance statistics
    #     Parameters:
    #     freq (int): 4 for qtrly, 12 for monthly, 252 for daily
    #     df (dataframe): dataframe of returns specified by the frequency
    #     """
    #     returns_df = self.value_df.groupby('Date').sum()['value'])
    #     stats = pd.DataFrame(columns = ['Model'], index=['Ann. Returns', 'Ann. Vol', 
    #                      'Sharpe','Downside Vol', 'Sortino', 'Max 12M Drawdown'])
    #     df2 = returns_df.dropna()
    #     stats.iloc[0] = df2.mean() * freq
    #     stats.iloc[1] = df2.std() * sqrt(freq)
    #     stats.iloc[2] = (df2.mean()*freq) / (df2.std()*sqrt(freq))
    #     lpm = df2.clip(upper=0) 
    #     stats.iloc[3] = (lpm.loc[(lpm != 0)]).std() * sqrt(freq)
    #     stats.iloc[4] = (df2.mean()*freq) / ((lpm.loc[(lpm != 0)]).std()*sqrt(freq))
    #     stats.iloc[5] = ((1+df2).rolling(freq).apply(np.prod, raw=False) - 1).min()
        
    #     return stats

    def run(self, start_dt, end_dt):
        """Runs strategy on portfolio until end_dt"""
        price_df = self.exchange.get_price_df(start_dt, end_dt)
        # if pd.to_datetime(end_dt) <= self.portfolio.ds:
        #     raise ValueError('end_dt must be after current portfolio date!')

        # pbar = tqdm(total = (pd.to_datetime(end_dt) - pd.to_datetime(start_dt))
        for row in price_df.itertuples():
            date = row[0]
            if self.exchange.is_end_of_month(date):
                print(date)
                actual_weights = (self.strategy
                                       .compute_actual_weights(self.portfolio))
                target_weights = (self.strategy
                                       .compute_target_weights(self.portfolio))
                 # TODO: consider this code below for exchange or broker class
                trades = self.strategy.calculate_trades(self.portfolio)
                for ticker in trades.keys():
                    side = 'buy' if trades[ticker] > 0 else 'sell'
                    self.portfolio.place_order(ticker, side,
                                                np.abs(trades[ticker]),
                                                'market', self.exchange)
            self.update_value_df()
            pbar.update()
        pbar.close()
        
        print(f'backtest_finished. It is now {end_dt}')
        
        #     self.portfolio.pass_time()
        #     if self.exchange.is_end_of_month(self.portfolio.ds):
        #         pdb.set_trace()
        #         actual_weights = (self.strategy
        #                               .compute_actual_weights(self.portfolio))
        #         target_weights = (self.strategy
        #                               .compute_target_weights(self.portfolio))
        #         # TODO: consider this code below for exchange or broker class
        #         trades = self.strategy.calculate_trades(self.portfolio)
        #         for ticker in trades.keys():
        #             side = 'buy' if trades[ticker] > 0 else 'sell'
        #             self.portfolio.place_order(ticker, side,
        #                                        np.abs(trades[ticker]),
        #                                        'market', self.exchange)
        #     self.update_value_df()
        #     pbar.update()
        # pbar.close()
       
        # self.plot_total_return()
        #print(self.calc_performance(252))
