import numpy as np

from .crypto_trading_env import CryptoTradingEnv, Actions


class CryptoEnv(CryptoTradingEnv):

    def __init__(self, df, window_size, frame_len, start_budget=100000):
        super().__init__(df, window_size, frame_len, start_budget)
        self.trade_fee_bid_percent = 0.01  # unit
        self._unit = 1  # units of btc

        self._quantity = 0  # positive quantity

    def _process_data(self, verbose=False):
        repartion_train_test = 0.8

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
        indice_rep = int(np.floor(prices.shape[0] * repartion_train_test))
        if verbose:
            print(f"Signal rows : {len(signal_features)}")
            print(f"Signal columns : {len(signal_features[0])}")
            print(signal_features, end='\n\n')
        train_price = prices[:indice_rep]
        test_price = prices[indice_rep:]
        train_features = signal_features[:indice_rep]
        test_features = signal_features[indice_rep:]
        return train_price, test_price, train_features, test_features
        # return prices, signal_features

    def _get_local_state(self):
        return self._quantity

    def _calculate_reward(self, action, terminal=False):
        if(self.training):
            prices = self.train_prices
        else:
            prices = self.test_prices
        next_price = prices[int(self._current_tick+1)]
        current_price = prices[int(self._current_tick)]
        #print(f"Current price : {current_price} USD")

        if terminal:
            # etat terminal -> on revend tout au prix du marché pour avoir notre profit
            s_fees = 1 if self._quantity < 0 else -1
            transation_amount = self._quantity * self._unit * current_price * \
                (1 + s_fees * self.trade_fee_bid_percent)   # Sell or buy everything we need/can
            self._total_profit += transation_amount
            reward = self._total_profit
            return reward

        else:
            if action == Actions.Buy.value:  # Buy
                self._quantity += 1
                current_transaction_amount = -self._unit * \
                    current_price * (1+self.trade_fee_bid_percent)
                self._total_profit += current_transaction_amount

            elif action == Actions.Sell.value:  # Sell
                self._quantity -= 1
                current_transaction_amount = self._unit * \
                    current_price * (1-self.trade_fee_bid_percent)
                self._total_profit += current_transaction_amount

            s_fees = - 1 if self._quantity < 0 else 1
            reward = self._quantity * self._unit * \
                (next_price - current_price) * \
                (1 + s_fees * self.trade_fee_bid_percent)
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
        end_tick = self._end_tick
        if(self.training):
            prices = self.train_prices
        else:
            prices = self.test_prices
        diff_price = prices[start_tick + 2:end_tick + 2] - \
            prices[start_tick + 1:end_tick + 1]
        profit = 0.
        quantity = 0
        for i, d in enumerate(diff_price):
            s_diff = -1 if d < 0 else 1
            quantity += self._unit * s_diff
            profit += self._unit * s_diff * prices[start_tick + i + 1]
        profit += quantity * self._unit * prices[end_tick + 1]
        return profit
