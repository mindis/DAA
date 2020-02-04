import pandas as pd
import numpy as np
from math import sqrt

def returns(df, horizon):
    """Calculates multiplicative returns of all series in a data frame"""
    return (df / df.shift(periods=horizon)) - 1    

def perf_stats(returns_df, freq, stratname):
    """ Calculate performance statistics
    Parameters:
    freq (int): 4 for qtrly, 12 for monthly, 252 for daily
    df (dataframe): dataframe of returns specified by the frequency
    """    
    stats = pd.DataFrame(columns = [stratname], index=['Ann. Returns', 'Ann. Vol', 
                     'Sharpe','Downside Vol', 'Sortino', 'Max 12M Drawdown'])
    df2 = returns_df.dropna()
    stats.iloc[0] = df2.mean() * freq
    stats.iloc[1] = df2.std() * sqrt(freq)
    stats.iloc[2] = (df2.mean()*freq) / (df2.std()*sqrt(freq))
    lpm = df2.clip(upper=0) 
    stats.iloc[3] = (lpm.loc[(lpm != 0)]).std() * sqrt(freq)
    stats.iloc[4] = (df2.mean()*freq) / ((lpm.loc[(lpm != 0)]).std()*sqrt(freq))
    stats.iloc[5] = ((1+df2).rolling(freq).apply(np.prod, raw=False) - 1).min()
    return stats
