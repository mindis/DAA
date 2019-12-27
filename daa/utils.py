import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from math import sqrt

# equally weighted, user-defined weights, mean-variance optimization,
# inverse variance portfolio (risk parity), hierarchical risk parity 
CONSTRUCTION = ['EqualWeight','CustomWeight', 'MVO', 'IVP', 'HRP']
REBALANCE = ['Limit','TripleBarrier', 'Monthly', 'Quarterly', 'Annual']

def send_four():
    return 4

def returns(df, horizon):
    """Calculates multiplicative returns of all series in a data frame"""
    return (df / df.shift(periods=horizon)) - 1    

def perf_stats(df, freq):
    """ Calculate Performance statistics
    Parameters:
    freq (int): 4 for qtrly, 12 for monthly, 252 for daily
    Returns:
    tuple: TODO(baogorek): needs description
    """
    df2 = df.dropna()
    ret = df2.mean() * freq
    vol = df2.std() * sqrt(freq)
    sharpe = ret / vol
    lpm = df2.clip(upper=0)
    down_vol = lpm.loc[(lpm != 0).any(axis=1)]
    sortino = ret / down_vol
    return ret, vol, sharpe, down_vol, sortino

class Exchange:
    
    price_df = None
    
    def __init__(self, price_filepath):
        """Takes csv of closing prices and simulates exchange"""
        self.price_df = pd.read_csv(price_filepath, index_col=0,
                                     parse_dates=True)
    
    def get_price_df(self):
        prices = self.price_df
        return prices    
    
    def get_price(self, ticker, ds):
        time_bar = self.price_df.index == pd.to_datetime(ds.date())
        price = float(self.price_df.loc[time_bar][ticker].values[0])
        print(price)
        return price

class Portfolio:
    def __init__(self, start_ds):
        self.id = 1
        self.ds = pd.to_datetime(start_ds)
        self.cash_balance = 0
        self.orders = []
        self.trade_id = 0

        self.trade_blotter = {}
        self.positions = defaultdict(lambda: 0)

    def get_trade_blotter(self):
        df = pd.DataFrame.from_dict(self.trade_blotter, orient='index')
        df.index.name = 'trade_id'
        return df

    def get_positions_df(self):
        """Returns a series indexed by ticker"""
        positions_df = pd.DataFrame()
        blotter_df = self.get_trade_blotter()
        if blotter_df.shape[0] > 0:
            positions_df = blotter_df.groupby(['ticker']).sum()['quantity'] 
        return positions_df

    def pass_time(self, units): # TODO: Assuming hours are always units
        for h in range(units):
            self.ds += datetime.timedelta(hours=1)
            print(self.ds)
            if self.ds.hour == 16: # Execute all orders when the market closes
                while len(self.orders) > 0:
                    order = self.orders.pop()
                    exchange = order[4]
                    ticker = order[0]
                    quantity = order[2]
                    side = order[1]
                    price = exchange.get_price(ticker, self.ds)
                    total_value = price * quantity
                    print("Before trade------------------------------")
                    print(self.get_trade_blotter())
                    print(self.get_positions_df())
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
                        #TODO: Steve mentions shares would be cleaner here
                        if quantity > self.get_positions_df().loc[ticker]:
                            raise ValueError('Not enough shares!')

                        self.trade_id += 1
                        self.cash_balance += total_value
                        self.trade_blotter[self.trade_id] = {
                                               'ds': self.ds, 'ticker': ticker,
                                    'quantity': -quantity,
                                    'price': price}
                    print("After trade------------------------------")
                    print(self.get_trade_blotter())
                    print(self.get_positions_df())


    def add_cash(self, amount):
        self.cash_balance += amount 

    def place_order(self, ticker, side, quantity, order_type, exchange):
        self.orders.append((ticker, side, quantity, order_type, exchange))
    
class Strategy:
    
    def __init__(self, prices, positions, strategy, num_assets, ID):
        
        """Main class to calculate trades based on strategy rules.
        
        Parameters
        ----------
        prices: pandas.DataFrame
            Matrix of prices. 
        potions: pandas.Series
            Matrix of positions.
        strategy: dictionary
            With keys 'name' (strategy algorithm)
            and 'allocation' (weighting scheme).
        num_assets: int
            Number of assets in modeled portfolio
        """
        self.id = ID # Customer unique ID
        self.N = num_assets 
        
        # Dataframe of asset prices with date index
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("The passed matrix is not a pandas.DataFrame!")
        else:
            if not isinstance(prices.index, pd.DatetimeIndex):
                raise ValueError("The passed matrix has not the proper index.")
        self.dates = prices.index
        self.prices = prices.values
        
        # Current portfolio positions
        if not isinstance(positions, pd.Series):
            raise ValueError("The passed matrix is not a pandas.Series.")
        # else: #TODO: Check if position tickers are in self.prices
        self.positions = positions
        
        # Strategy dictionary
        if not isinstance(strategy, dict):
            raise ValueError("The strategy is a dictionary.")
        if 'construction' not in strategy.keys():
            raise ValueError("Specify the strategy name!")
        if 'rebalance' not in strategy.keys():
            raise ValueError("Specify the weight distribution algorithm!")
        if strategy['construction'] not in CONSTRUCTION:
            raise NotImplementedError("Try again: Not a strategy")
        if strategy['rebalance'] not in REBALANCE:
            raise NotImplementedError("Try again: not a rebalancing frequency")
        self.strategy = strategy
        
        
