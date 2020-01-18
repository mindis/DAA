import pandas as pd

class Portfolio:
    """Portfolio is a snapshot in time of all positions."""
    def __init__(self, start_ds, exchange, cash, ID):
        """Initialize a Portfolio

        self.ds (DateTime): The current time in which Portfolio exists
        """
        self.id = ID
        self.exchange = exchange 
        self.price_df = exchange.get_price_df()
        self.price_row = self._get_row_from_date(start_ds)
        self.ds = pd.to_datetime(self.price_df.index[self.price_row])
        self.cash_balance = cash
        self.orders = []  # TODO: move this property to another class
        self.trade_id = 0 
        self.trade_blotter = {}

    def _get_row_from_date(self, start_ds):
        """From character string date representation (YYYY-MM-DD) to DF row"""
        price_df2 = self.price_df.reset_index()
        return price_df2.loc[price_df2.Date == start_ds].index[0]

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

    def get_prices(self):
        """Returns dictionary of prices at the current date"""
        return self.exchange.get_price_df().loc[self.ds].to_dict()

    def place_order(self, ticker, side, quantity, order_type, exchange):
        self.orders.append((ticker, side, quantity, order_type, exchange))
        buys = [order for order in self.orders if order[1] == 'buy']
        # Buy the most expensive asset first TODO: this is doing nothing here
        buys = sorted(buys, key=lambda order: -self.get_prices()[order[0]])
        # Sell the most expensive asset first TODO: this is doing nothing here
        sells = [order for order in self.orders if order[1] == 'sell']
        sells = sorted(sells, key=lambda order: -self.get_prices()[order[0]])
        # sells get popped off list first so sells happen before buys
        self.orders = buys + sells
        
    def cash_balance(self):
        return self.cash_balance
