
import pandas as pd
import numpy as np
from math import sqrt

raw_prices = pd.read_csv("C:\devl\DAA\DAA_RawPriceData_12.17.19.csv")

# select month-end dates
dates = pd.to_datetime(raw_prices.Date)
eom = dates[dates.dt.is_month_end]

# TODO: buy and hold
# Monthly S&P returns -> month end dates

# TODO 60 / 40
