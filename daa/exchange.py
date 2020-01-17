import pandas as pd

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
