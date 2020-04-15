"""
TESA MODEL: 4.13.20

@author: halls35
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

#Read data
df = pd.read_csv('tesa_data.csv', index_col=0, parse_dates=True).sort_index()
sector_names = list(df.columns)[:10]

# feature transformations
def pct_returns(df, end_dt, beg_dt):
    """Calculates percentage returns of all series in a data frame
        between two dates"""
    return (df.shift(periods=end_dt) / df.shift(periods=beg_dt)) - 1

def absolute_diff(df, end_dt, beg_dt):
    """Calculates absolute differences of all series in a data frame
        between two dates"""
    return df.shift(periods=end_dt) - df.shift(periods=beg_dt)

def rolling_ann_vol(df, window, ann_factor):
    """Calculates rolling annualized standard deviation given a 
        lookback window"""
    returns = df.pct_change()
    return returns.rolling(window).std() * np.sqrt(ann_factor)

# response variables
response_prices = df.iloc[:,:10]
# 21-day forward returns (shift back)
horizon = 21
response = pct_returns(response_prices, -horizon, 0).dropna()

# predictor variables
variables_diff = ['TwosTens', 'FivesThirties'] # zList of vars for 1st diff
prices_pct_chgs = df[df.columns.difference(variables_diff)]
prices_abs_diff = df[variables_diff]

momentum = pct_returns(prices_pct_chgs, 21, 252).dropna()
momentum = momentum.add_suffix('_MOMO')
reversal = pct_returns(prices_pct_chgs, 1, 21).dropna()
reversal = reversal.add_suffix('_1MR')
vol = rolling_ann_vol(prices_pct_chgs, 60, 252).dropna()
vol = vol.add_suffix('_VOL')

merge1 = pd.merge(momentum, reversal, left_on='Dates', right_on='Dates')
predictors = pd.merge(merge1, vol, left_index=True, right_index=True)

merged = pd.merge(response, predictors, left_index=True, right_index=True)
Y = merged.iloc[:,:10]
X = merged.iloc[:,10:]

# Describe predictor set
predictor_stats = predictors.describe()

# Model Parameters
k = 125 # number of neighbors
lb = 15*252 # lookback period (15 years)
wgt = 'distance' # uniform or distance method for KNN prediction

## Loop for predictions
n = len(Y) - lb - horizon
predictions = dict()
dts = X.index[-n:]
for s in range(Y.shape[1]):
    results = np.zeros((n,1))
    for i in range(n):
        X_train = X.iloc[i:lb+i,:]
        Y_train = Y.iloc[i:lb+i,s]
        model = KNeighborsRegressor(k, weights=wgt)
        model.fit(X_train, Y_train)
        X_test = np.asarray(X.iloc[lb+i+horizon,:]).reshape(1,-1)
        prediction = model.predict(X_test)
        results[i] = prediction
        name = sector_names[s]
        results_df = pd.DataFrame(results, index=dts, columns=[name])
        predictions[name] = results_df

## Build backtest
all_results_df = pd.concat(predictions.values(), axis=1, sort=True)
ranked_df = all_results_df.rank(axis=1)
fwd_returns = response_prices.pct_change().shift(-1)
fwd_returns = fwd_returns.add_suffix('_1d')
fwd_returns = pd.merge(fwd_returns, ranked_df, left_index=True, right_index=True)
fwd_returns = fwd_returns.iloc[:,:10].to_numpy()

# Weighting Schemes
ew = ranked_df.replace(list(range(1,11)), 1.0/10).to_numpy()

top_bottom_3 = ranked_df.replace([8, 9, 10], 0.2)
top_bottom_3 = top_bottom_3.replace([4, 5, 6, 7], 0.10)
top_bottom_3 = top_bottom_3.replace([1, 2, 3], 0.0).to_numpy()

top_3 = ranked_df.replace([8, 9, 10], 0.2)
top_3 = top_3.replace(list(range(1,8)), 0.4/7).to_numpy()

bottom_3 = ranked_df.replace([1, 2, 3], 0.0)
bottom_3 = bottom_3.replace(list(range(4,11)), 1.0/7).to_numpy()

# Portfolio Returns
port_ew = (fwd_returns * ew).sum(axis=1)
port_top_bottom_3 = (fwd_returns * top_bottom_3).sum(axis=1)
port_top_3 = (fwd_returns * top_3).sum(axis=1)
port_bottom_3 = (fwd_returns * bottom_3).sum(axis=1)

index_ew = np.cumprod(1 + port_ew)
index_top_bottom_3 = np.cumprod(1 + port_top_bottom_3)
index_top_3 = np.cumprod(1 + port_top_3)
index_bottom_3 = np.cumprod(1 + port_bottom_3)

# Performance Statistics
ann_return_ew = port_ew.mean()*252
ann_return_top_bottom_3 = port_top_bottom_3.mean()*252
ann_return_top_3 = port_top_3.mean()*252
ann_return_bottom_3 = port_bottom_3.mean()*252

ann_vol_ew = port_ew.std()*np.sqrt(252)
ann_vol_top_bottom_3 = port_top_bottom_3.std()*np.sqrt(252)
ann_vol_top_3 = port_top_3.std()*np.sqrt(252)
ann_vol_bottom_3 = port_bottom_3.std()*np.sqrt(252)

Sharpe_ew = ann_return_ew / ann_vol_ew
Sharpe_top_bottom_3 = ann_return_top_bottom_3 / ann_vol_top_bottom_3
Sharpe_top_3 = ann_return_top_3 / ann_vol_top_3
Sharpe_bottom_3 = ann_return_bottom_3 / ann_vol_bottom_3

print ("Equally Weighted Sharpe = {}".format(round(Sharpe_ew,2)))
print ("Top/Bottom 3 Sharpe = {}".format(round(Sharpe_top_bottom_3,2)))
print ("Top 3 Only Sharpe = {}".format(round(Sharpe_top_3,2)))
print ("Bottom 3 Only Sharpe = {}".format(round(Sharpe_bottom_3,2)))

# Plot total returns
plt.plot(dts, index_ew, label='EW')
plt.plot(dts, index_top_bottom_3, label='Top/Bottom 3')
plt.plot(dts, index_top_3, label='Top 3')
plt.plot(dts, index_bottom_3, label='Bottom 3')
plt.ylabel('Cumulative Returns')
plt.xlabel('Dates')
plt.axis('tight')
plt.legend()
plt.title('Equally Weighted v. KNN Models')
plt.show()










