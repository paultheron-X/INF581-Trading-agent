import numpy as np

from .crypto_trading_env import CryptoTradingEnv, Actions


class CryptoEnv(CryptoTradingEnv):

    def __init__(self, df, window_size, frame_len, start_budget=100000):
        super().__init__(df, window_size, frame_len, start_budget)
        self.trade_fee_bid_percent = 0.01  # unit
        self._unit = 1  # units of btc

        self._long = 0  # positive quantity
        self._short = 0  # Positive -> amount that you short

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

    def _get_local_state(self):
        return self._short, self._long

    def _calculate_reward(self, action, terminal=False):
        next_price = self.prices[int(self._current_tick+1)]
        current_price = self.prices[int(self._current_tick)]
        print(current_price)

        if terminal:
            print('salt')
            # etat terminal -> on revend tout au prix du marché pour avoir notre profit
            positive_transation_amount = self._long * self._unit * current_price * \
                (1-self.trade_fee_bid_percent)   # Sell everything I own
            negative_transation_amount = -self._short * self._unit * current_price * \
                (1+self.trade_fee_bid_percent)   # Buy everything I short
            self._total_profit += positive_transation_amount + negative_transation_amount
            reward = self._total_profit
            return reward

        else:
            if action == Actions.Buy:  # Buy
                self._long += 1
                current_transaction_amount = -self._unit * \
                    current_price * (1+self.trade_fee_bid_percent)
                self._total_profit += current_transaction_amount
            elif action == Actions.Sell:  # Sell
                self._short += 1
                current_transaction_amount = self._unit * \
                    current_price * (1-self.trade_fee_bid_percent)
                self._total_profit += current_transaction_amount

            diff = (self._long - self._short)
            fees = - self.trade_fee_bid_percent if diff < 0 else self.trade_fee_bid_percent
            reward = diff * self._unit * \
                (next_price - current_price) * (1 + fees)

            return reward

    # Je mets dans CryptoTrading la MaJ du buget et de la quantité
    def _update_profit_reward(self, action, terminal=False):
        instant_reward = self._calculate_reward(
            action=action, terminal=terminal)
        self._total_reward += instant_reward
        return instant_reward

    def max_possible_profit(self):
        # la fonction est à réécrire, mais dans l'idée, c'est ça
        # sachant qu'on ne prend pas en compte les fees :
        # il faudrait compter en benef les moindres augmentations entre 2 temps
        start_tick = self._start_tick + self._padding_tick
        profit = 0.
        for i in range(start_tick, self._end_tick + 1):
            for j in range(i + 1, self._end_tick + 1):
                d = self.prices[j] / self.prices[i]
                if d > profit:
                    profit = d
        return profit
