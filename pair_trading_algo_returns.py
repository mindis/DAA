"""
Created on Mon Jan 27 17:48:18 2020

@author: halls35
"""
import numpy as np
import pandas as pd
import  time as t

import matplotlib.pyplot as plt

from datetime import datetime

import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tia.bbg import LocalTerminal

import datetime
datetime.datetime.strptime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

start_timer = t.time()

## Load data using BBG API.
#SPX_sectors=pd.DataFrame()
#rep1 = LocalTerminal.get_historical(['S5INFT Index', 'S5ENRS Index', 
#                                     'S5FINL Index', 'S5CONS Index', 
#                                     'S5HLTH Index', 'S5COND Index', 
#                                     'S5INDU Index', 'S5UTIL Index', 
#                                     'S5MATR Index', 'S5TELS Index'], 
#                                    ['PX_LAST'], start='9/25/1989',             
#                                    period='DAILY')
#SPX_sectors=rep1.as_frame()
#
## Rename columns
#cols = ['Tech', 'Energy', 'Fin', 'Cons', 'Hlth', 'Disc', 'Indu', 
#        'Utils', 'Matr', 'Tels']
#SPX_sectors.columns = cols
## Save file as pickle
#SPX_sectors.to_pickle("C:\devl\EquityResearch\data\SPX_sectors")

# START HERE if data is pickled. 
# Unpickle file.
SPX_sectors = pd.read_pickle("C:\devl\EquityResearch\data\SPX_sectors")

cols = ['Tech', 'Energy', 'Fin', 'Cons', 'Hlth', 'Disc', 'Indu', 
        'Utils', 'Matr', 'Tels']

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


scores, pvalues, pairs = find_cointegrated_pairs(SPX_sectors)

# Heatmap to show the p-values of the cointegration test
# between each pair of stocks
import seaborn
m = [0,0.2,0.4,0.6,0.8,1]
seaborn.heatmap(pvalues, xticklabels=cols, 
                yticklabels=cols, cmap='RdYlGn_r' 
                , mask = (pvalues >= 0.98)
                )
plt.show()

# Plot price ratio of Industrials/Materials
S1 = SPX_sectors['Indu']
S2 = SPX_sectors['Utils']
score, pvalue, _ = coint(S1, S2)
print(pvalue)
ratios = S1 / S2
ratios.plot(figsize=(15,7))
plt.axhline(ratios.mean())
plt.legend(['Price Ratio'])
plt.show()

def zscore(series):
    return (series - series.mean()) / np.std(series)

zscore(ratios).plot(figsize=(15,7))
plt.axhline(zscore(ratios).mean(), color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Ratio z-score', 'Mean', '+1', '-1'])
plt.show()

# Create price ratios
ratios = S1 / S2
oos_len = len(ratios) - 252*3
train = ratios[:oos_len]
test = ratios[oos_len:]
S1_train = S1.iloc[:oos_len]
S2_train = S2.iloc[:oos_len]
S1_test = S1.iloc[oos_len:]
S2_test = S2.iloc[oos_len:]

# Create labels
ratios_mavg5 = train.rolling(window=5,
                               center=False).mean()

ratios_mavg60 = train.rolling(window=60,
                               center=False).mean()

std_60 = train.rolling(window=60,
                        center=False).std()

zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
plt.figure(figsize=(15,7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)
plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])
plt.ylabel('Ratio')
plt.show()

# Rolling lookback test
window = 60
signal = 1.5 # z-score to buy or sell ratio
std_rolling = train.rolling(window=window,center=False).std()
std_rolling.name = 'std rolling'

# Compute the z score for each day
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
zscore_60_5.name = 'z-score'

plt.figure(figsize=(15,7))
zscore_60_5.plot()
plt.axhline(0, color='black')
plt.axhline(signal, color='red', linestyle='--')
plt.axhline(-signal, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1.5', '-1.5'])
plt.show()

# Plot the ratios and buy and sell signals from z score
plt.figure(figsize=(15,7))

train[window:].plot()
buy = train.copy()
sell = train.copy()

buy[zscore_60_5>-signal] = 0
sell[zscore_60_5<signal] = 0
buy[window:].plot(color='g', linestyle='None', marker='^')
sell[window:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()

# Plot the prices and buy and sell signals from z score
plt.figure(figsize=(18,9))
S1_train[window:].plot(color='b')
S2_train[window:].plot(color='c')
buyR = 0*S1_train.copy()
sellR = 0*S1_train.copy()

# When buying the ratio, buy S1 and sell S2
buyR[buy!=0] = S1_train[buy!=0]
sellR[buy!=0] = S2_train[buy!=0]
# When selling the ratio, sell S1 and buy S2 
buyR[sell!=0] = S2_train[sell!=0]
sellR[sell!=0] = S1_train[sell!=0]

buyR[window:].plot(color='g', linestyle='None', marker='^')
sellR[window:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,min(S1_train.min(),S2_train.min()),max(S1_train.max(),S2_train.max())))

plt.legend(['Indu','Utils', 'Buy Signal', 'Sell Signal'])
plt.show()

# Trade using a simple strategy
def trade(S1, S2, window1, window2, signal):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0
    
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    
    S1_rets = S1.pct_change()
    S2_rets = S2.pct_change()
    
    # Simulate trading
    # Start with no money and no positions
    rets = pd.Series()
    side = 0
    
    for i in range(1,len(ratios)):
        # Sell short the spread if the z-score is > 1
        if zscore[i] > signal :
            dret = pd.Series(S1_rets[i]-S2_rets[i], index=[S1_rets.index[i]])
            rets = pd.concat([rets,dret])
            side = -1
        # Buy long the spread if the z-score is < 1
        elif zscore[i] < -signal or side == 1 :
            dret = pd.Series(-S1_rets[i]+S2_rets[i], index=[S1_rets.index[i]])
            rets = pd.concat([rets,dret])
            side = 1
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.5:
            side = 0 
            dret = pd.Series(0, index=[S1_rets.index[i]])
            rets = pd.concat([rets,dret])
        # Else stay short the spread if you were short previously    
        elif side == -1:
            dret = pd.Series(S1_rets[i]-S2_rets[i], index=[S1_rets.index[i]])
            rets = pd.concat([rets,dret])
        # Else stay long the spread if you were long previously    
        elif side == 1:
            dret = pd.Series(-S1_rets[i]+S2_rets[i], index=[S1_rets.index[i]])
            rets = pd.concat([rets,dret])
            side = 1          
        else: 
            side = 0 
            dret = pd.Series(0, index=[S1_rets.index[i]])
            rets = pd.concat([rets,dret])
            
    daily_returns = rets.rename('DailyReturns')
    
    return daily_returns

daily_returns = trade(S1_train, S2_train, 5, 32, 1.5)

ann_mean = daily_returns.mean() * 252
ann_vol = daily_returns.std() * np.sqrt(252)
sharpe = ann_mean/ann_vol

cum_returns = pd.Series(np.cumprod(1 + daily_returns.values), 
                                   index=daily_returns.index).rename('TR')


# Find the window length 0-254 
# that gives the highest returns using this strategy

scores = pd.Series()
for l in range(20,255,5):
    
    rets = trade(S1_train, S2_train, 5, l, 1.5)
    ann_mean = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_mean/ann_vol
    
    score = pd.Series(sharpe, index=[l])    
    scores = pd.concat([scores,score])
    
#length_scores = [trade(S1_train, 
#                S2_train, 5, l, 1.5) 
#                for l in range(20,255,5)]

best_length = np.argmax(scores)

print ('Best window length:', best_length)

#
#plt.figure(figsize=(15,7))
#plt.plot(cum_returns.index, cum_returns.values)
#plt.ylabel('TR')
#plt.title('Cumulative Returns: Indu v. Utils')
#plt.show()
#
#money, daily_returns, cum_returns = trade(S1_test, S2_test, 5, 32, 1.5)
#
##OOS Test
#plt.figure(figsize=(15,7))
#plt.plot(cum_returns.index, cum_returns.values)
#plt.ylabel('TR')
#plt.title('OOS Test Cumulative Returns: Indu v. Matr')
#plt.show()
#
## Find the returns for test data
## using what we think is the best window length
#length_scores2 = [trade(S1_test, 
#                  S2_test, 5, l, 1.5) 
#                  for l in range(30,50)]
#
#print (best_length, 'day window:', length_scores2[best_length])
#
## Find the best window length based on this dataset, 
## and the returns using this window length
#best_length2 = np.argmax(length_scores2)
#print (best_length2, 'day window:', length_scores2[best_length2])
#
#plt.figure(figsize=(15,7))
#plt.plot(length_scores)
#plt.plot(length_scores2)
#plt.xlabel('Window length')
#plt.ylabel('Score')
#plt.legend(['Training', 'Test'])
#plt.show()
#
#
