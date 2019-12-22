
def get_last_business_day_of_month(df):
    #TODO
    pass

def returns(df, horizon):
    """Calculates multiplicative returns of all series in a data frame"""
    return (df / df.shift(periods=horizon)) - 1    

def perf_stats(df, freq):
    """ Calculate Performance statistics

    Parameters:
    freq (int): 4 for qtrly, 12 for monthly, 252 for daily

    Returns:
    tuple: TODO(baogorek): needs description
    """
    df2 = df.dropna()
    ret = df2.mean() * freq
    vol = df2.std() * sqrt(freq)
    sharpe = ret / vol
    lpm = df2.clip(upper=0)
    down_vol = lpm.loc[(lpm != 0).any(axis=1)]
    sortino = ret / down_vol
    return ret, vol, sharpe, down_vol, sortino
