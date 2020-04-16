# ETF Factor Model Portfolio Optimization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

from tia.bbg import LocalTerminal

# Get monthly data from Bloomberg 
tickers = ['SPY', 'VLUE', 'QUAL', 'SIZE', 'MTUM', 'USMV']
factor_names = ['Value', 'Quality', 'Size', 'Momentum', 'MinVol']
req2 = LocalTerminal.get_historical(['SPY Equity', 'VLUE Equity', 'QUAL Equity',
                                     'SIZE Equity', 'MTUM Equity', 'USMV Equity'], 
                                    ['PX_LAST'], start='1/2/1995',             
                                    period='DAILY', ignore_security_error=1)

ETF_Prices = req2.as_frame()
ETF_Prices.columns = tickers
ETF_Prices = ETF_Prices.dropna()

ETF_Prices.to_csv('Factor_ETF_Prices.csv')
SPY = ETF_Prices.iloc[:,0]
SPY_returns = SPY.pct_change().dropna()
Factors = ETF_Prices.iloc[:,1:]
Factor_returns = Factors.pct_change().dropna()

# Loop to solve for portfolio weights each period 
backtest_wgts = []

return_lb = 6*21 #(eight months)
covar_lb = 2*252 # 252 trading days
rows = len(ETF_Prices.index) - covar_lb + 1 # length of loop
dates = ETF_Prices.index[covar_lb-1:] # datetime index

for i in range(rows):
    mu1 = expected_returns.mean_historical_return(Factors.iloc[covar_lb-return_lb+i:covar_lb+i])
    mu = np.clip(mu1, 0.01, 0.20)
    S = risk_models.sample_cov(Factors.iloc[i:covar_lb+i,:])
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.30))
    try: 
        raw_weights = ef.max_sharpe()
        wgts = list(ef.clean_weights().values())
    except: 
        wgts = [0.2] * 5
    backtest_wgts.append(wgts)

wgts_df = pd.DataFrame(backtest_wgts, index=dates, columns=factor_names)
wgts_arr = np.array(backtest_wgts[:-1])
returns_arr = (Factor_returns.iloc[-len(wgts_df)+1:]).to_numpy()

model_returns = (wgts_arr * returns_arr).sum(axis=1)
SPY_returns = SPY_returns[-len(model_returns):]
dts = SPY_returns.index[-len(model_returns):]

model_index = np.cumprod(1 + model_returns)
SPY_index = np.cumprod(1 + SPY_returns)

# Performance Statistics
model_ann_return = model_returns.mean()*252
model_ann_vol = model_returns.std()*np.sqrt(252)
model_Sharpe = model_ann_return / model_ann_vol

SPY_ann_return = SPY_returns.mean()*252
SPY_ann_vol = SPY_returns.std()*np.sqrt(252)
SPY_Sharpe = SPY_ann_return / SPY_ann_vol

print ("Model Sharpe = {}".format(round(model_Sharpe,2)))
print ("SPY Sharpe = {}".format(round(SPY_Sharpe,2)))
print ("Model Growth of 10k = ${}".format(round(model_index[-1]*10000,2)))
print ("SPY Growth of 10k = ${}".format(round(SPY_index[-1]*10000,2)))

# Plot total returns
plt.plot(dts, model_index, label='Model')
plt.plot(dts, SPY_index, label='SPY')
plt.ylabel('Cumulative Returns')
plt.xlabel('Dates')
plt.axis('tight')
plt.legend()
plt.title('Model vs. SPY Cumulative Returns')
plt.show()
