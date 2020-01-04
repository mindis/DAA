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
        
    def cash_balance(self):
        return self.cash_balance
    
class Backtest:
    
    def __init__(self, portfolio, strategy, dates, ID):
        
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
        self.id = ID # placeholder for customer unique ID
        self.initial_cash = portfolio.cash_balance
        self.dates = dates
        self.start_dt = self.dates.min()
        self.end_dt = self.dates.max()

        # Dataframe of asset prices with date index
        self.price_df = portfolio.exchange.get_price_df() 
        self.prices = self.price_df.values
        self.num_assets = self.price_df.shape[1]
        
        # Current portfolio positions
        self.positions_df = portfolio.get_positions_df()
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
        return init_weights
    
    def run(self):

        cash = self.initial_cash
        dates = self.dates
        num_assets = self.num_assets
        
        # Always start equally weighted
        # Assumes initial cash put to work on close of first day
        price1 = self.prices[0]
        shares = (cash/num_assets)/price1 # assumes fractional shares        
        
        # TODO: Should we create initial orders here? 
        
        # start the backtest
        for i, day in enumerate(dates):

            if self.strategy['name'] == 'EqualWeight':
                
                # TODO:
                # Execute any trades in order blotter from previous day.
                # Then calculate current portfolio weights.
                # curr_weights = portfolio.get_positions_df().
                # Calculate target weights.
                # Compare target weights to current weights.
                # Calculate number of trades.
                # Submit trades to Exchange for execution at day + 1
                # Next day
                print(day)

class Strategy:
    def __init__(self, strategy):
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

