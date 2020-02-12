import pandas as pd
import pdb

class Portfolio:
    """Portfolio is a snapshot in time of all positions."""
    def __init__(self, date, exchange, cash, ID):
        """Initialize a Portfolio

        self.ds (DateTime): The current time in which Portfolio exists
        self.exchange (Object): Price exchange
        self.cash_balance (Float) = liquidity in portfolio
        self.id (integer) = unique user ID
        """
        self.id = ID
        self.ds = pd.to_datetime(date)
        self.exchange = exchange 
        self.cash_balance = float(cash)
        self.orders = []  # TODO: move this property to another class
        self.trade_id = 0 
        self.trade_blotter = {}
    
    def add_cash(self, amount):
        self.cash_balance += amount
    
    def cash_balance(self):
        return self.cash_balance

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
            update = pd.DataFrame(blotter_df.groupby(['ticker'])
                                            .sum()['quantity'])
            positions_df = pd.concat([positions_df, update], sort=False)
            for idx in range(len(positions_df)):
                ticker = positions_df.index[idx]
                if ticker == 'Cash':
                    price = 1.
                else:
                    price = self.exchange.get_price(self.ds, ticker)
                positions_df.loc[positions_df.index[idx], 'price'] = price
                
            positions_df['value'] = positions_df['price'] * positions_df['quantity']
            positions_df['wgt'] = positions_df['value'] / positions_df['value'].sum()
        return positions_df
    
    def place_order(self, ticker, side, quantity, order_type, exchange):
        self.orders.append((ticker, side, quantity, order_type, exchange))
        buys = [order for order in self.orders if order[1] == 'buy']
        sells = [order for order in self.orders if order[1] == 'sell']
        self.orders = buys + sells
    
    def execute_orders(self):
        while len(self.orders) > 0:
            order = self.orders.pop()
            ticker = order[0]
            side = order[1]                   
            quantity = order[2]
            exchange = order[4]
            price = exchange.get_price(self.ds, ticker)
            total_value = price * quantity
            if side == 'buy':
                if total_value > self.cash_balance:
                    quantity -= 1
                    # TODO 
                    print(self.get_positions_df())
                    print(f"Couldn't buy {quantity} of {ticker} on {self.ds}")
                    print(f"Tried to buy {total_value}s worth with {self.cash_balance}")
                    total_value = price * quantity

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
                
