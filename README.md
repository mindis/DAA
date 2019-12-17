# DAA
Dynamic Asset Allocation Model
Dynamic Asset Allocation: Project Plan
1)	Obtain asset class index and tradable, liquid ETF price data. 
a)	Daily or intra-day is best; can always calculate weekly or monthly returns
b)	Broad market-based asset classes to start (Equities, Govt bonds, Corporate bonds, MBS, Munis) 
c)	TODO: automate data retrieval from open, free source
2)	Develop module to calculate simple portfolio constructions
a)	80/20, 70/30, 60/40, etc. (equity, bonds)
b)	Equally-weighted
c)	Annual, quarterly, and monthly rebalancing to illustrate the value of rebalancing (at least historically)
3)	Develop module to forecast long-term asset class returns using simple models
a)	CAPE Shiller P/E for equity returns
b)	Current yields for fixed income
c)	Example: https://www.factorresearch.com/research-global-pension-funds-the-coming-storm
4)	Develop module to estimate covariances across asset classes
a)	Simple, annualized standard deviation
b)	Exponentially-weighted standard deviation
c)	GARCH model
i)	https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/
5)	Develop module for advanced portfolio construction techniques
a)	Mean-variance optimization
i)	https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
b)	Risk-parity, also known as the inverse variance portfolio (IVP)
c)	Hierarchical Risk Parity (HRP)
i)	Google: advances in financial machine learning pdf (pgs. 221-230)
6)	Build front-end that allows user to toggle different portfolio weights across asset classes or input return, volatility and correlation estimates into our portfolio construction engine that will output optimized portfolios
There is enormous value in providing these tools and education inexpensively for the masses. This is all my team does as institutional asset allocators. We could start by building something that sits on top of Robin Hood, the execution platform.  
If we realize some success, we could expand by:
1)	Overlaying Bryan Jordan’s credit cycle framework (although I think this is too simplistic and could be improved).
2)	Build an economic nowcast (hot marketing term in quant finance). GDP comes out quarterly. A lot of top banks and research firms are trying to produce real-time GDP estimates using higher frequency data. The implications are straightforward. Accelerating GDP tends to correlate with earnings growth and equity price appreciation.
3)	Incorporate sectors and individual stocks/bonds. This would take some more bodies in my opinion. To obtain, clean, and maintain a comprehensive database of security price data is a heavy lift. NW Investments has 50 IT people and can’t figure it out. 
