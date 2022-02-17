import numpy as np

from .crypto_trading_env import CryptoTradingEnv
from .trading_env import Actions, Positions


class CryptoEnv(CryptoTradingEnv):

    def __init__(self, df, window_size, frame_len, start_budget=100000):
        super().__init__(df, window_size, frame_len, start_budget)
        self.trade_fee_bid_percent = 0.01  # unit
        self.bid = 10 #units of btc

    def _process_data(self):

        prices = self.df.loc[:, 'close'].to_numpy()
        features = self.df.loc[:,
                               [
                                   "open", "high",
                                   "low", "close",
                                   "Volume BTC", "Volume USD"
                               ]
                               ].to_numpy()

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.c_[features, diff]

        return prices, signal_features

    def _calculate_reward(self, action):
        current_price = self.prices[int(self._current_tick)]
        last_trade_price = self.prices[int(self._last_trade_tick)]
        price_diff = current_price - last_trade_price

        
        if action == 1: # buy
            return - price_diff
        elif action ==2: #sell
            return price_diff
        else:
            return 0

    def _update_profit(self, action):
        
        # faux, il faut aller voir avec comme la fonction calculate reward
        p = self.prices[int(self._current_tick)]
        if action == 1: # buy
            self._budget += self.bid * p * (1 - self.trade_fee_bid_percent)
            self._quantity += self.bid
            self._total_profit = self._quantity * p + self._budget
        elif action ==2: #sell
            self._budget -= self.bid * p * (1 - self.trade_fee_bid_percent)
            self._quantity -= self.bid
            self._total_profit = self._quantity * p + self._budget

    def max_possible_profit(self):
        # la fonction est à réécrire, mais dans l'idée, c'est ça
        # sachant qu'on ne prend pas en compte les fees :
        # il faudrait compter en benef les moindres augmentations entre 2 temps
        start_tick = self._start_tick + self._padding_tick
        last_trade_tick = start_tick - 1
        profit = 1.

        for i in range(start_tick, self._end_tick + 1):
            for j in range(i + 1, self._end_tick + 1):
                d = self.prices[j] / self.prices[i]
                if d > profit:
                    profit = d

        return profit * self.start_budget
