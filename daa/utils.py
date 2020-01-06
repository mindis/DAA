import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from math import sqrt

def returns(df, horizon):
    """Calculates multiplicative returns of all series in a data frame"""
    return (df / df.shift(periods=horizon)) - 1    

def perf_stats(returns_df, freq, stratname):
    """ Calculate performance statistics
    Parameters:
    freq (int): 4 for qtrly, 12 for monthly, 252 for daily
    df (dataframe): dataframe of returns specified by the frequency
    """    
    stats = pd.DataFrame(columns = [stratname], index=['Ann. Returns', 'Ann. Vol', 
                     'Sharpe','Downside Vol', 'Sortino', 'Max 12M Drawdown'])
    df2 = returns_df.dropna()
    stats.iloc[0] = df2.mean() * freq
    stats.iloc[1] = df2.std() * sqrt(freq)
    stats.iloc[2] = (df2.mean()*freq) / (df2.std()*sqrt(freq))
    lpm = df2.clip(upper=0) 
    stats.iloc[3] = (lpm.loc[(lpm != 0)]).std() * sqrt(freq)
    stats.iloc[4] = (df2.mean()*freq) / ((lpm.loc[(lpm != 0)]).std()*sqrt(freq))
    stats.iloc[5] = ((1+df2).rolling(freq).apply(np.prod, raw=False) - 1).min()
    return stats

class Exchange:
    
    price_df = None
    
    def __init__(self, price_filepath):
        """Takes csv of closing prices and simulates exchange"""
        self.price_df = pd.read_csv(price_filepath, index_col=0,
                                     parse_dates=True)
    
    def get_price_df(self):
        prices = self.price_df
        return prices  
    
    def is_end_of_month(self, date):
        date_ts = pd.to_datetime(date)
        all_dates = self.price_df.index
        mmyyyy = all_dates.strftime('%Y%m')
        dates_dict = all_dates.groupby(mmyyyy)
        eom_dates = [max(dates_dict[month]) for month in dates_dict.keys()] 
        return date_ts in eom_dates
    
    def get_price(self, ticker, ds):
        time_bar = self.price_df.index == pd.to_datetime(ds.date())
        if ticker not in self.price_df.columns:
            raise KeyError("Try again: Ticker not found")
        price = float(self.price_df.loc[time_bar][ticker].values[0])
        return price

# TODO: fix start time
class Portfolio:
    """Portfolio is a snapshot in time of all positions."""
    def __init__(self, start_ds, exchange, cash, ID):
        """Initialize a Portfolio

        self.ds (DateTime): The current time in which Portfolio exists
        """
        self.id = ID
        self.exchange = exchange 
        self.price_df = exchange.get_price_df()
        self.price_row = 0 
        self.ds = pd.to_datetime(self.price_df.index[self.price_row])
        self.cash_balance = cash
        self.orders = []  # TODO: move this property to another class
        self.trade_id = 0 
        self.trade_blotter = {}

    def get_trade_blotter(self):
        df = pd.DataFrame.from_dict(self.trade_blotter, orient='index')
        df.index.name = 'trade_id'
        return df

    def get_positions_df(self):
        """Returns a dataframe indexed by ticker"""
        positions_df = pd.DataFrame({'quantity': self.cash_balance, 
                                     'value': self.cash_balance},
                                    index=['Cash'])
        positions_df.index.name = 'ticker'
        blotter_df = self.get_trade_blotter()
        if blotter_df.shape[0] > 0:
            positions_df = pd.concat([positions_df, 
                                      pd.DataFrame(blotter_df.groupby(['ticker'])
                                            .sum()['quantity'])])
            
            for idx in range(len(positions_df)):
                ticker = positions_df.index[idx]
                if ticker == 'Cash':
                    price = 1.
                else:
                    price = self.exchange.get_price(ticker, self.ds)
                positions_df.loc[positions_df.index[idx], 'price'] = price
                
            positions_df['value'] = positions_df['price'] * positions_df['quantity']
            positions_df['wgt'] = positions_df['value'] / positions_df['value'].sum()
        return positions_df
    
    def pass_time(self):
        self.price_row += 1
        self.ds = pd.to_datetime(self.price_df.index[self.price_row])
        while len(self.orders) > 0:
            order = self.orders.pop()
            ticker = order[0]
            exchange = order[4]                    
            quantity = order[2]
            side = order[1]
            price = exchange.get_price(ticker, self.ds)
            total_value = price * quantity
            if side == 'buy':
                if total_value > self.cash_balance:
                    raise ValueError('Not enough cash!')

                self.trade_id += 1
                self.cash_balance -= total_value 
                self.trade_blotter[self.trade_id] = {
                                       'ds': self.ds, 'ticker': ticker,
                                       'quantity': quantity,
                                       'price': price}

            elif side == 'sell':
                if quantity > self.get_positions_df().loc[ticker,'quantity']:
                    raise ValueError('Not enough shares!')
                self.trade_id += 1
                self.cash_balance += total_value
                self.trade_blotter[self.trade_id] = {
                                       'ds': self.ds, 'ticker': ticker,
                                       'quantity': -quantity,
                                       'price': price}


    def add_cash(self, amount):
        self.cash_balance += amount 

    def place_order(self, ticker, side, quantity, order_type, exchange):
        self.orders.append((ticker, side, quantity, order_type, exchange))
        buys = [order for order in self.orders if order[1] == 'buy']
        sells = [order for order in self.orders if order[1] == 'sell']
        self.orders = buys + sells # buys get popped off list first
        
    def cash_balance(self):
        return self.cash_balance
