import numpy as np

class Strategy1:
    def __init__(self, exchange, schedule):
        self.exchange = exchange
        self.price_df = exchange.get_price_df()
        self.schedule = 'M'

    def compute_actual_weights(self, portfolio):
        tickers = ['SPX_Index', 'EAFE_Index', 'EM_Index']
        total = portfolio.cash_balance
        value_dict = {'Cash': portfolio.cash_balance} 
        purchased_tickers = portfolio.get_positions_df().index.to_list()  

        for ticker in purchased_tickers:
            value_dict[ticker] = (portfolio.get_positions_df()
                                           .loc[ticker]['value'])
            total += value_dict[ticker]

        for ticker in [t for t in tickers if t not in purchased_tickers]:
            value_dict[ticker] = 0.0

        for ticker in value_dict.keys():
            value_dict[ticker] = value_dict[ticker] / total

        return value_dict

    def compute_target_weights(self, portfolio):
        return {'Cash': 0., 'SPX_Index': .25, 'EAFE_Index': .25, 'EM_Index': .50}

    def calculate_trades(self, porfolio):
        actual_weights = self.compute_actual_weights(portfolio)
        target_weights = self.compute_target_weights(portfolio)

        tickers = set(actual_weights.keys()) - {'Cash'}

        prices = {'Cash': 1.}
        for ticker in tickers:
            prices[ticker] = price_df.loc[portfolio.ds][ticker]

        total = portfolio.get_positions_df()['value'].sum() + portfolio.cash_balance

        delta_weights = {}
        shares = {}
        for ticker in tickers:
            delta_weights[ticker] = target_weights[ticker] - actual_weights[ticker]
            shares[ticker] = np.floor(total * delta_weights[ticker] / prices[ticker])

        return shares
