import numpy as np

from .trading_env import CryptoTradingEnv, Actions, Positions


class CryptoEnv(CryptoTradingEnv):

    def __init__(self, df, window_size, frame_len, start_budget=100000):
        super().__init__(df, window_size, frame_len, start_budget)
        self.trade_fee_bid_percent = 0.01  # unit

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
        signal_features = np.c_((features, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price

        if action != 0:
            return - price_diff * action

        return price_diff

    def _update_profit(self, action):
        p = self.prices[self._current_tick]
        self._budget -= action * p * (1 - self.trade_fee_bid_percent)
        self._quantity += action
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
