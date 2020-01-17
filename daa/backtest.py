import pandas as pd
import numpy as np

class Backtest:
    """Facilitates backtesting strategy on porfolio given exchange"""
    def __init__(self, portfolio, exchange, strategy):
        self.portfolio = portfolio
        self.exchange = exchange
        self.strategy = strategy

    def run(self, end_dt):
        """Runs strategy on portfolio until end_dt"""
        if pd.to_datetime(end_dt) <= self.portfolio.ds:
            raise ValueError('end_dt must be after current portfolio date!')

        while self.portfolio.ds < pd.to_datetime(end_dt):
            self.portfolio.pass_time()
            print(self.portfolio.ds)
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
        print("backtest_finished")
