import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from math import sqrt

# equally weighted, user-defined weights, mean-variance optimization,
# inverse variance portfolio (risk parity), hierarchical risk parity 
NAME = ['EqualWeight','CustomWeight', 'MVO', 'IVP', 'HRP']
REBALANCE = ['Limit','TripleBarrier', 'Monthly', 'Quarterly', 'Annual']
LOOKBACK = ['1YR', '2YR', '3YR', '5YR', '10YR']


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
        if ticker not in self.price_df.columns:
            raise KeyError("Try again: Ticker not found")
        price = float(self.price_df.loc[time_bar][ticker].values[0])
        print(price)
        return price

class Portfolio:
    def __init__(self, start_ds, exchange, cash, ID):
        self.id = ID
        self.ds = pd.to_datetime(start_ds)
        self.cash_balance = cash
        self.orders = []
        self.trade_id = 0
        self.exchange = exchange
        self.trade_blotter = {}
        self.positions = defaultdict(lambda: 0)

    def get_trade_blotter(self):
        df = pd.DataFrame.from_dict(self.trade_blotter, orient='index')
        df.index.name = 'trade_id'
        return df

    def get_positions_df(self):
        """Returns a dataframe indexed by ticker"""
        positions_df = pd.DataFrame()
        blotter_df = self.get_trade_blotter()
        if blotter_df.shape[0] > 0:
            positions_df = pd.DataFrame(blotter_df.groupby(['ticker'])
                                        .sum()['quantity'])
        
        for idx in range(len(positions_df)):
            ticker = positions_df.index[idx]
            price = self.exchange.get_price(ticker, self.ds)
            positions_df.loc[positions_df.index[idx], 'price'] = price
            
        positions_df['value'] = positions_df['price'] * positions_df['quantity']
        positions_df['wgt'] = positions_df['value'] / positions_df['value'].sum()
        return positions_df

    def pass_time(self, units): # TODO: Assuming hours are always units
        for h in range(units):
            self.ds += datetime.timedelta(hours=1)
            print(self.ds)
            if self.ds.hour == 16: # Execute all orders when the market closes
                while len(self.orders) > 0:
                    order = self.orders.pop()
                    ticker = order[0]
                    exchange = order[4]                    
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
    
class Backtest:
    
    def __init__(self, exchange, portfolio, strategy, dates):
        
        """Main class to calculate trades based on strategy rules.
        
        Parameters
        ----------
        exchange: object containing dates and prices
            Matrix of prices. 
        potions: pandas.Series
            Matrix of positions.
        strategy: dictionary
            With keys 'name' (strategy algorithm)
            and 'allocation' (weighting scheme).
        num_assets: int
            Number of assets in modeled portfolio
        dates: DatetimeIndex
            index of user-defined dates for the backtest
        """
        self.id = portfolio.ID # Customer unique ID  
        self.dates = dates
        self.start_dt = self.dates.min()
        self.end_dt = self.dates.max()

        # Dataframe of asset prices with date index
        self.price_df = exchange.get_price_df() 
        if not isinstance(self.price_df, pd.DataFrame):
            raise ValueError("The passed matrix is not a pandas.DataFrame.")
        else:
            if not isinstance(self.price_df.index, pd.DatetimeIndex):
                raise ValueError("The passed matrix has not the proper index.")
        self.prices = self.price_df.values
        self.num_assets = self.price_df.shape[1]
        
        # Current portfolio positions
        self.positions_df = portfolio.get_positions_df()
        if not isinstance(self.positions_df, pd.DatFrame):
            raise ValueError("The passed matrix is not a pandas.DataFrame.")
        
        # Strategy dictionary
        if not isinstance(strategy, dict):
            raise ValueError("The strategy is not a dictionary.")
        if 'name' not in strategy.keys():
            raise ValueError("Specify the strategy name!")
        if 'rebalance' not in strategy.keys():
            raise ValueError("Specify the weight distribution algorithm.")
        if strategy['name'] not in NAME:
            raise NotImplementedError("Try again: Not a strategy")
        if strategy['rebalance'] not in REBALANCE:
            raise NotImplementedError("Try again: not a rebalancing method")
        if strategy['lookback'] not in LOOKBACK:
            raise NotImplementedError("Try again: not a lookback window")
        self.strategy = strategy
        
    def get_initial_weights(self):

        # initialize weights
        if self.strategy['name'] == 'EqualWeight':
            init_weights = 1/self.num_assets           
        elif self.strategy['name'] == 'CustomWeight':
            #TODO: user-defined
            init_weights = 1/self.num_assets
        elif self.strategy['name'] == 'MVO':
            #TODO: Mean-variance optimization
            init_weights = 1/self.num_assets
        elif self.strategy['name'] == 'IVP':
            #TODO: Inverse variance portfolio
            init_weights = 1/self.num_assets
        elif self.strategy['name'] == 'HRP':
            #TODO: Hierarchical Risk Parity (Ch. 16: Advances in Financial ML)
            init_weights = 1/self.num_assets
        
