import pandas as pd
import numpy as np
import math
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
raw_prices = pd.read_csv('.\data\PriceData.csv')

## Buy and HOld v. Rebalancing
# Initialize assumptions
init_equity = 10000
wgt_SPX = 0.6
wgt_AGG = 0.4

# Select month-end dates
raw_prices['Date'] = pd.to_datetime(raw_prices.Date)
Dates = pd.to_datetime(raw_prices.Date)
eom = Dates[Dates.dt.is_month_end]
eom_prices = raw_prices[raw_prices['Date'].isin(eom)].reset_index(drop=True)

# Calculate returns
def returns(df,horizon):
    return (df/df.shift(periods=horizon)) - 1   
rets = returns(eom_prices.iloc[:,1:],1)

# Buy and Hold Portfolio
eom_SPX = eom_prices['SPX_Index'].reset_index(drop=True)
eom_AGG = eom_prices['AGG_Index'].reset_index(drop=True)
shares_SPX = math.floor(init_equity*wgt_SPX/eom_SPX[0]) # rounddown
shares_AGG = math.floor(init_equity*wgt_AGG/eom_AGG[0]) # rounddown
equity_SPX = shares_SPX*eom_SPX
equity_AGG = shares_AGG*eom_AGG
equity_total_BH = equity_SPX+equity_AGG
rets_BH = returns(equity_total_BH,1)

# 60/40 portfolio
ret_SPX = rets['SPX_Index'].fillna(0)
ret_AGG = rets['AGG_Index'].fillna(0)
port_ret = wgt_SPX*ret_SPX+wgt_AGG*ret_AGG
cum_ret = np.cumprod(1+port_ret)
equity_total_RB = init_equity*cum_ret
equity_delta = equity_total_BH-equity_total_RB
rets_RB = returns(equity_total_RB,1)

# Performance statistics
def perf_stats(df,freq): #freq = 4 for qtrly, 12 for monthly, 252 for daily
    df2 = df.dropna()
    ret = df2.mean()*freq
    vol = df2.std()*sqrt(freq)
    sharpe = ret/vol
    lpm = df2.clip(upper=0)
    down_vol = lpm.loc[(lpm!=0).any(axis=1)]
    sortino = ret/down_vol
    return (ret, vol, sharpe, down_vol, sortino)

# Plot results
fig, ax = plt.subplots()
locator = mdates.AutoDateLocator(minticks=5, maxticks=7)
formatter = mdates.AutoDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.plot(eom, equity_total_BH, 'k', label='Buy-Hold')
ax.plot(eom, equity_total_RB, 'b', label='Rebal')
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
ax.plot(eom, equity_delta, 'r', label='Delta')
ax.set_xlabel('Date')
ax.set_ylabel('Buy-Hold less Monthly Rebal')
ax.legend()
plt.title('Cumulative Delta between Buy-Hold and Monthly Rebal')
plt.show()

