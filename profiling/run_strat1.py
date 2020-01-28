#!/usr/bin/env python

"""
Create the profile out_profile with the following line at the shell:

python -m cProfile -o out_profile run_strat1.py

Then, enter the Profile statistics browser with 

python -m pstats out_profile

sort time
stats 10

sort cumulative
stats 5

tottime is the total time spent in the function alone.
cumtime is the total time spent in the function plus all
functions that this function called

"""
import os
import sys

sys.path.append('..')

from daa.strategy import Strategy1, Strategy2
from daa.utils import *
from daa.portfolio import *
from daa.exchange import *
from daa.backtest import *


def main():
    start_dt = '1999-01-04'
    end_dt = '2001-12-31'
    exchange = Exchange(os.path.join('../daa/data/PriceData.csv'))
    portfolio = Portfolio(start_dt, exchange, 1000000, 0)
    strategy = Strategy1(exchange, 'M')
    backtest = Backtest(portfolio, exchange, strategy)
    backtest.run(end_dt)
    portfolio.get_positions_df()

if __name__ == "__main__":
    os.chdir("..")
    main()
