import numpy as np
import pandas as pd

class Exchange:
    def __init__(self, price_filepath):
        """Takes csv of closing prices and simulates exchange"""
        self.price_df = self._create_price_df(price_filepath)

    def _create_price_df(self, price_filepath):
        price_df = pd.read_csv(price_filepath)
        price_df['Date'] = pd.to_datetime(price_df.Date)
        price_df['mmyyyy'] = price_df.Date.apply(lambda dt: dt.strftime('%Y%m'))
        price_df['lead_mmyyyy'] = price_df.mmyyyy.shift(-1)
        price_df['eobm'] = price_df.mmyyyy != price_df.lead_mmyyyy
        price_df.iloc[-1]['eobm'] = np.nan  # TODO:Need another way for last dt
        price_df.set_index('Date', inplace=True)
        return price_df

    def get_price(self, ds, ticker):
        if ticker not in self.price_df.columns:
            raise KeyError("Try again: Ticker not found")
        price = float(self.price_df.loc[ds][ticker])
        return price
