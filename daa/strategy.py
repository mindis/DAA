import numpy as np
import pandas as pd
from collections import defaultdict
import pdb

class Strategy1:
    def __init__(self, exchange, schedule):
        self.exchange = exchange
        self.schedule = 'M'

    def compute_actual_weights(self, portfolio):
        tickers = ['SPX_Index', 'EAFE_Index', 'EM_Index']
        total = 0
        value_dict = {} 
        purchased_tickers = portfolio.get_positions_df().index.to_list()  

        for ticker in purchased_tickers:
            value_dict[ticker] = (portfolio.get_positions_df()
                                           .loc[ticker]['value'])
            total += value_dict[ticker]

        for ticker in [t for t in tickers if t not in purchased_tickers]:
            value_dict[ticker] = 0.0

        for ticker in value_dict.keys():
            value_dict[ticker] = value_dict[ticker] / total

        return value_dict

    def compute_target_weights(self, portfolio):
        target_dict = defaultdict(lambda: 0.)
        target_dict['Cash'] = .25,
        target_dict['SPX_Index'] = .25
        target_dict['EAFE_Index'] = .25
        target_dict['EM_Index'] =.50
        return target_dict

    def calculate_trades(self, portfolio):
        #pdb.set_trace()
        value = portfolio.get_positions_df()['value'].sum()
        portfolio_shares = portfolio.get_positions_df()['quantity']
        target_weights = self.compute_target_weights(portfolio)

        tickers = set(target_weights.keys()) - {'Cash'}
        
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

    #def calculate_trades(self, portfolio):
    #    
    #    actual_weights = self.compute_actual_weights(portfolio)
    #    target_weights = self.compute_target_weights(portfolio)

    #    tickers = set(target_weights.keys()) - {'Cash'}
    #    
    #    prices = {'Cash': 1.}
    #    for ticker in tickers:
    #        prices[ticker] = self.exchange.get_price(portfolio.ds, ticker)
    #    total = portfolio.get_positions_df()['value'].sum()

    #    delta_weights = {}
    #    shares_to_trade = {}
    #    for ticker in tickers:
    #        delta_weights[ticker] = target_weights[ticker] - actual_weights[ticker]
    #        shares = total * delta_weights[ticker] / prices[ticker]
    #        shares = np.sign(shares) * np.floor(np.abs(shares))
    #        if np.abs(shares) > 0:
    #            shares_to_trade[ticker] = shares

    #    return shares_to_trade

class Strategy2:
    def __init__(self, exchange, schedule):
        self.exchange = exchange
        self.schedule = schedule 

    def compute_actual_weights(self, portfolio):
        tickers = ['SPX_Index', 'AGG_Index']
        total = 0
        value_dict = {} 
        purchased_tickers = portfolio.get_positions_df().index.to_list()  

        for ticker in purchased_tickers:
            value_dict[ticker] = (portfolio.get_positions_df()
                                           .loc[ticker]['value'])
            total += value_dict[ticker]

        for ticker in [t for t in tickers if t not in purchased_tickers]:
            value_dict[ticker] = 0.0

        for ticker in value_dict.keys():
            value_dict[ticker] = value_dict[ticker] / total

        return value_dict

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
        
        # TODO (baogorek): reminder that tickers must be typed 3 times
        if tr_cape > 16:
           target_dict['SPX_Index'] = np.exp(-(tr_cape - 16)**2 / 40)
           target_dict['AGG_Index'] = 1. - target_dict['SPX_Index']

        return target_dict

    def calculate_trades(self, portfolio):
        #pdb.set_trace()
        value = portfolio.get_positions_df()['value'].sum()
        portfolio_shares = portfolio.get_positions_df()['quantity']
        target_weights = self.compute_target_weights(portfolio)

        tickers = set(target_weights.keys()) - {'Cash'}
        
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

    
class Strategy3:
    def __init__(self, exchange, schedule, ticker_dict, lookback):
        self.exchange = exchange
        self.price_df = exchange.price_df
        self.ma_df = self.price_df.rolling(lookback).mean()
        self.tickers = list(ticker_dict.keys())
        self.momentum_flags = list(ticker_dict.values())
        self.lookback = lookback
        self.schedule = 'D'

    def compute_actual_weights(self, portfolio):
        total = 0
        value_dict = {} 
        purchased_tickers = portfolio.get_positions_df().index.to_list()  

        for ticker in purchased_tickers:
            value_dict[ticker] = (portfolio.get_positions_df()
                                           .loc[ticker]['value'])
            total += value_dict[ticker]

        for ticker in [t for t in self.tickers if t not in purchased_tickers]:
            value_dict[ticker] = 0.0

        for ticker in value_dict.keys():
            value_dict[ticker] = value_dict[ticker] / total

        return value_dict

    def compute_target_weights(self, portfolio):
        target_dict = defaultdict(lambda: 0.)

        momos = [t[0] for t in zip(self.tickers, self.momentum_flags) if t[1]]
        no_momos = [t for t in self.tickers if t not in momos]

        for ticker in momos:
            price = self.exchange.get_price(portfolio.ds, ticker)
            if price > self.ma_df.loc[portfolio.ds, ticker]:
                target_dict[ticker] = 1.0
            else:
                target_dict[ticker] = 0.0
        momo_sum = sum(target_dict.values())
        if momo_sum > 1.0: 
            target_dict.update((x, y / momo_sum) for x, y in target_dict.items())
        elif momo_sum == 0.0:
            for ticker in no_momos:
                target_dict[ticker] = 1.0 / len(no_momos)

        return target_dict

    def calculate_trades(self, portfolio):
        #pdb.set_trace()
        value = portfolio.get_positions_df()['value'].sum()
        portfolio_shares = portfolio.get_positions_df()['quantity']
        target_weights = self.compute_target_weights(portfolio)

        tickers = set(target_weights.keys()) - {'Cash'}
        
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
#TODO: see backtester TODO on itertuples
