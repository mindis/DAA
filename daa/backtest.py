import pandas as pd
import numpy as np
from math import sqrt

from tqdm import tqdm
import pdb

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
register_matplotlib_converters()

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
        # TODO: is drop duplicates the best way to ensure cash isn't added twice?
        self.value_df = pd.concat([self.value_df, update], 
                                  sort=False).drop_duplicates()

    def plot_total_return(self):
        plt.plot(self.value_df.groupby('Date').sum()['value'])
        plt.title('Cumulative Growth of Initial Investment')
        plt.show()
        
    def calc_performance(self, freq):
        """ Calculate performance statistics
        Parameters:
        freq (int): 4 for qtrly, 12 for monthly, 252 for daily
        df (dataframe): dataframe of returns specified by the frequency
        """
        total_value = self.value_df.groupby('Date')['value'].sum()
        returns = total_value.pct_change().dropna()
        stats = pd.DataFrame(columns = ['Model'], index=['Ann. Returns', 'Ann. Vol', 
                          'Sharpe','Downside Vol', 'Sortino', 'Max 12M Drawdown'])
        stats.iloc[0] = returns.mean() * freq
        stats.iloc[1] = returns.std() * sqrt(freq)
        stats.iloc[2] = (returns.mean()*freq) / (returns.std()*sqrt(freq))
        lpm = returns.clip(upper=0) 
        stats.iloc[3] = (lpm.loc[(lpm != 0)]).std() * sqrt(freq)
        stats.iloc[4] = (returns.mean()*freq) / ((lpm.loc[(lpm != 0)]).std()*sqrt(freq))
        stats.iloc[5] = ((1+returns).rolling(freq).apply(np.prod, raw=False) - 1).min()

        return stats

    def run(self, end_dt):
        """Runs strategy on portfolio until end_dt"""
        if pd.to_datetime(end_dt) <= self.portfolio.ds:
            raise ValueError('end_dt must be after current portfolio date!')

        price_df = self.exchange.price_df.loc[self.portfolio.ds:end_dt]
        timedelta = len(price_df)
        pbar = tqdm(total = timedelta)
        
        for row in price_df.itertuples():
            self.portfolio.ds = row[0] # Move the portfolio to the next business day
            if price_df.loc[self.portfolio.ds].eobm:
                trades = self.strategy.calculate_trades(self.portfolio)
                for ticker in trades.keys():
                    side = 'buy' if trades[ticker] > 0 else 'sell'
                    self.portfolio.place_order(ticker, side,
                                                  np.abs(trades[ticker]),
                                                  'market', self.exchange)
                self.portfolio.execute_orders()
            pbar.update()
            self.update_value_df()
        
        pbar.close()
        print(f'backtest_finished. It is now {end_dt}')
        self.plot_total_return()
        print(self.calc_performance(252))
