import pandas as pd
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from daa.utils import *

# User inputs
horizon = 1

# Load data
raw_prices = pd.read_csv('./data/PriceData.csv', index_col=0,
                         parse_dates=['Date'])
raw_prices.freq = pd.infer_freq(raw_prices.index)

## Buy and Hold v. Rebalancing
# Initialize assumptions
init_equity = 10000
weight_SPX = 0.6
weight_AGG = 0.4

# Select month-end dates
eom = raw_prices.asfreq('BM').index  # last business day of month
eom_prices = raw_prices[raw_prices.index.isin(eom)]

# TODO(baogorek): Allow user to specify daily, weekly, etc. to avoid confusion
horizon_returns = returns(eom_prices, horizon)  # Currently, 1-month return

# Strategy 1: 60/40 SPX/AGG Buy and Hold Portfolio (no rebalancing)
eom_SPX = eom_prices['SPX_Index']
eom_AGG = eom_prices['AGG_Index']

initial_shares_SPX = math.floor(init_equity * weight_SPX / eom_SPX[0])
initial_shares_AGG = math.floor(init_equity * weight_AGG / eom_AGG[0])
equity_SPX = initial_shares_SPX * eom_SPX
equity_AGG = initial_shares_AGG * eom_AGG
equity_total_buy_hold = equity_SPX + equity_AGG
returns_buy_hold = returns(equity_total_buy_hold, 1)

# Strategy: 60/40 SPX/AGG portfolio with monthly rebalancing
returns_SPX = horizon_returns['SPX_Index'].fillna(0)
returns_AGG = horizon_returns['AGG_Index'].fillna(0)
portfolio_returns = weight_SPX*returns_SPX + weight_AGG*returns_AGG
cumulative_returns = np.cumprod(1 + portfolio_returns)
equity_total_rebalance = init_equity * cumulative_returns
equity_delta = equity_total_buy_hold - equity_total_rebalance
returns_rebalance = returns(equity_total_rebalance, horizon)

# Plot results
fig, ax = plt.subplots()
locator = mdates.AutoDateLocator(minticks=5, maxticks=7)
formatter = mdates.AutoDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.plot(equity_total_buy_hold, 'k', label='Buy-Hold')
ax.plot(equity_total_rebalance, 'b', label='Rebalance')
ax.set_xlabel('Date')
ax.set_ylabel('Total Equity')
ax.legend()
plt.title('Growth of $10,000: 60/40 Buy-Hold v. Monthly Rebal')
plt.show()

fig, ax = plt.subplots()
locator = mdates.AutoDateLocator(minticks=5, maxticks=7)
formatter = mdates.AutoDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.plot(equity_delta, 'r', label='Delta')
ax.set_xlabel('Date')
ax.set_ylabel('Buy-Hold less Monthly Rebal')
ax.legend()
plt.title('Cumulative Delta between Buy-Hold and Monthly Rebal')
plt.show()
