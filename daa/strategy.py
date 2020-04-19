from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pdb

# TODO: separate strategies into other files
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


class Strategy(ABC):
    """An abstract class from which to create strategies as children"""
    def __init__(self, exchange, schedule):
        self.exchange = exchange
        self.price_df = exchange.price_df
        self.schedule = schedule 
        self.check_schedule_validity()

    def check_schedule_validity(self):
        if self.schedule not in ['D', 'M']:
            raise ValueError('Choose Daily "D" or Monthly "M" schedules')

    @abstractmethod
    def compute_target_weights(self, portfolio):
        target_dict = defaultdict(lambda: 0.)    
        return target_dict

    def calculate_trades(self, portfolio):
        value = portfolio.get_positions_df()['value'].sum()
        portfolio_shares = portfolio.get_positions_df()['quantity']
        target_weights = self.compute_target_weights(portfolio)

        tickers = set(target_weights.keys())
        tickers = tickers.union(portfolio.get_positions_df().index.tolist())
        tickers = tickers - {'Cash'}
        shares_to_trade = {}

        for ticker in tickers:
            price = self.exchange.price_df.loc[portfolio.ds][ticker]
            target_shares = np.floor(value * target_weights[ticker] / price)
            current_shares = 0
            
            if ticker in portfolio_shares.index.to_list():
                current_shares = portfolio_shares[ticker] 

            if target_shares != current_shares:
                shares_to_trade[ticker] = target_shares - current_shares

        return shares_to_trade


class Fixed_Weights(Strategy):
    def __init__(self, exchange, schedule, ticker_dict):
        super().__init__(exchange, schedule)
        
        self.ticker_dict = ticker_dict
        
    def compute_target_weights(self, portfolio):
        
        target_dict = self.ticker_dict
        
        return target_dict


class CAPE(Strategy):
    def __init__(self, exchange, schedule):
        super().__init__(exchange, schedule)

    def get_trcape(self, portfolio):
        """Compute Shiller's TRCape metric""" 
        # TODO: properly handle this data set
        trcape_df = pd.read_csv('data/trcape.csv', parse_dates=True,
                                index_col=0)
        trcape_df['yyyymm'] = trcape_df.index.to_period('M')
        # Merge tr_cape info in by month
        price_df = self.exchange.price_df.copy()
        price_df['yyyymm'] = price_df.index.to_period('M')
        price_df.reset_index(inplace=True)
        price_df = price_df.merge(trcape_df, on='yyyymm')
        price_df.set_index('Date', inplace=True)

        return price_df.loc[portfolio.ds].tr_cape

    def compute_target_weights(self, portfolio):
        target_dict = defaultdict(lambda: 0.)
        target_dict['Cash'] = 0.0
        target_dict['SPX_Index'] = 1.0 
        target_dict['AGG_Index'] = 0.0

        tr_cape = self.get_trcape(portfolio)
        
        if tr_cape > 16:
           target_dict['SPX_Index'] = np.exp(-(tr_cape - 16)**2 / 40)
           target_dict['AGG_Index'] = 1. - target_dict['SPX_Index']

        return target_dict

    
class Trend_Following(Strategy):
    """A trend following strategy based on moving average of len lookback"""
    def __init__(self, exchange, schedule, ticker_dict, lookback):
        super().__init__(exchange, schedule)

        self.ma_df = self.price_df.rolling(lookback).mean()
        self.tickers = list(ticker_dict.keys())
        self.momentum_flags = list(ticker_dict.values())
        self.lookback = lookback

    def compute_target_weights(self, portfolio):
        target_dict = defaultdict(lambda: 0.)    
        
        momos = [t[0] for t in zip(self.tickers, self.momentum_flags) if t[1]]
        no_momos = [t for t in self.tickers if t not in momos]
       
        for ticker in momos:
            price = self.exchange.get_price(portfolio.ds, ticker)
            if price > self.ma_df.loc[portfolio.ds, ticker]:
                target_dict[ticker] = 1.0
                for ticker in no_momos:
                    target_dict[ticker] = 0.0
            else:
                target_dict[ticker] = 0.0
        momo_sum = sum(target_dict.values())
        if momo_sum > 1.0: 
            target_dict.update((x, y / momo_sum) for x, y in target_dict.items())
            for ticker in no_momos:
                target_dict[ticker] = 0.0
        elif momo_sum == 0.0:
            for ticker in no_momos:
                target_dict[ticker] = 1.0 / len(no_momos)
        if np.abs(sum(target_dict.values()) - 1.0) > 0.001:
               raise ValueError("Weights do not sum to 1.")
        if min(target_dict.values()) < 0.0:
               raise ValueError("Security weight less than 0.")
                   
        return target_dict


class Minimum_Variance(Strategy):
    def __init__(self, exchange, schedule, tickers):
        super().__init__(exchange, schedule)

        self.tickers = tickers

        # TODO(baogorek): find solution to prices that don't change daily
        good_data = exchange.price_df[tickers].loc['1996-01-01':]
        self.returns = good_data.pct_change().dropna()

    def compute_target_weights(self, portfolio):
        pdb.set_trace()
        target_dict = defaultdict(lambda: 0.)

        Sigma = self.returns.loc[:portfolio.ds].cov().to_numpy()
        e = np.ones(Sigma.shape[0])
        Sigma_inv = np.linalg.inv(Sigma)

        w = np.matmul(Sigma_inv, e) / np.matmul(e, np.matmul(Sigma_inv, e))
        for ticker, weight in zip(self.tickers, w):
            target_dict[ticker] = weight

        return target_dict



class Max_Sharpe(Strategy):
    def __init__(self, exchange, schedule, tickers, lookback):
        super().__init__(exchange, schedule)

        self.tickers = tickers
        self.lookback = lookback
        self.prices = self.exchange.price_df[tickers].dropna().loc['1996-01-02':]

    def compute_target_weights(self, portfolio):
        target_dict = defaultdict(lambda: 0.)

        prices = self.prices.reset_index()
        current_i = prices[prices['Date'] == portfolio.ds].index.values[0]
        local_df = prices.iloc[(current_i - self.lookback):current_i]
        local_df.set_index('Date', inplace=True)

        mu = expected_returns.mean_historical_return(local_df)
        S = risk_models.sample_cov(local_df)

        ef = EfficientFrontier(mu, S)
        target_dict = ef.max_sharpe()

        return target_dict
