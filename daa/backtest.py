import pandas as pd
import numpy as np

from tqdm import tqdm

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

    def run(self, end_dt):
        """Runs strategy on portfolio until end_dt"""
        if pd.to_datetime(end_dt) <= self.portfolio.ds:
            raise ValueError('end_dt must be after current portfolio date!')

        pbar = tqdm(total = (pd.to_datetime(end_dt) - self.portfolio.ds).days)
        while self.portfolio.ds < pd.to_datetime(end_dt):
            self.portfolio.pass_time()
            if self.exchange.is_end_of_month(self.portfolio.ds):
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
        self.plot_total_return()
