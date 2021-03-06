import pandas as pd
import numpy as np
from math import sqrt

from tqdm import tqdm
import pdb

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

from daa.portfolio import Portfolio

pd.options.mode.chained_assignment = None
register_matplotlib_converters()

class Backtest:
    """Facilitates backtesting strategy on porfolio given exchange"""
    def __init__(self, exchange, strategy, start_dt, end_dt, start_cash=1E5):
        
        self.portfolio = Portfolio(start_dt, exchange, start_cash, 0)
        self.exchange = exchange
        self.strategy = strategy
        self.value_df = pd.DataFrame()
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.start_cash = start_cash
        if pd.to_datetime(end_dt) <= pd.to_datetime(start_dt):
            raise ValueError('end_dt must be after current portfolio date!')
        self.sp500_benchmark = self.run_sp500_benchmark()

    def update_value_df(self):
        update = self.portfolio.get_positions_df().reset_index()
        update['Date'] = self.portfolio.ds
        update.set_index('Date', inplace=True)
        self.value_df = pd.concat([self.value_df, update], sort=False)

    def plot_total_returns(self):
        plt.plot(self.value_df.groupby('Date').sum()['value'],
                 label = type(self.strategy).__name__,
                 linewidth=.8)
        plt.plot(self.sp500_benchmark, label='S&P500', linewidth='.8')
        ax = plt.gca()
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter
                                           (lambda x, 
                                            loc: "{:,}".format(int(x))))
        plt.legend()
        plt.title('Cumulative Growth of Initial Investment')
        plt.show()
        
    def plot_weights(self):
        tickers = list(set(self.value_df.ticker))
        for ticker in tickers:
            ticker_df = self.value_df.loc[self.value_df.ticker == ticker] 
            fig, ax = plt.subplots()
            ax.set_title(f'Portfolio Profile: {ticker}')
            ax.set_ylim(-.1, 4)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_ylabel(f'Weight of {ticker} in Portfolio')
            ax.step(ticker_df.index, ticker_df['wgt'], color='green',
                    linewidth=.8)
            ax2 = ax.twinx()
            ax2.grid(b=False)
            ax2.set_ylabel(f'Total Return Index: {ticker}')
            ax2.plot(ticker_df.index, ticker_df.price, linewidth=.8)
            
        
    def calc_performance(self):
        """ Calculate performance statistics
        Parameters:
        freq (int): 4 for qtrly, 12 for monthly, 252 for daily
        df (dataframe): dataframe of returns specified by the frequency
        """
        total_value = self.value_df.groupby('Date')['value'].sum()
        strategy_returns = total_value.pct_change().dropna()
        benchmark_returns = self.sp500_benchmark.pct_change().dropna()
        returns = pd.concat([strategy_returns, benchmark_returns], axis=1)

        stats = pd.DataFrame(columns = [type(self.strategy).__name__,
                                        'Benchmark'],
                             index=['Ann. Returns', 'Ann. Vol', 'Sharpe',
                                    'Downside Vol', 'Sortino'])
        freq = 252
        stats.iloc[0, :] = returns.mean().values * freq
        stats.iloc[1, :] = returns.std().values * sqrt(freq)
        stats.iloc[2, :] = ((returns.mean().values*freq)
                            / (returns.std().values*sqrt(freq)))
        stats.iloc[3, :] = returns.apply(lambda col:
                                         np.std(col[col < 0])).values * sqrt(freq)
        stats.iloc[4, :] = stats.iloc[0, :] / stats.iloc[3, :]
        # TODO(baogorek): Implement drawdown
        return stats

    def run_sp500_benchmark(self):
        price_df = self.exchange.price_df.loc[self.portfolio.ds:self.end_dt]
        spx_prices = price_df['SPY']
        initial_qty = self.start_cash / spx_prices[0]  # partial shares allowed
        benchmark_value = spx_prices * initial_qty
        return benchmark_value               
       
    def run(self):
        """Runs strategy on portfolio until end_dt"""
        price_df = self.exchange.price_df.loc[self.portfolio.ds:self.end_dt]
        timedelta = len(price_df)
        pbar = tqdm(total = timedelta)
        
        for row in price_df.itertuples():
            self.portfolio.ds = row[0] # Move to the next business day

            trade_ind = True # schedule == 'D'
            if self.strategy.schedule == 'M':
                trade_ind = price_df.loc[self.portfolio.ds].eobm

            if trade_ind:
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
        print(f'backtest_finished. It is now {self.end_dt}')
        self.plot_total_returns()
        self.plot_weights()
        print(self.calc_performance())

