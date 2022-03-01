import numpy as np
from torch import threshold

from .crypto_trading_env import CryptoTradingEnv, Actions


class CryptoEnv(CryptoTradingEnv):

    def __init__(self, df, window_size, frame_len, start_budget=100000):
        super().__init__(df, window_size, frame_len, start_budget)
        self.trade_fee_bid_percent = 0.01  # unit
        self._unit = 1  # units of btc
        self._quantity = 0  # positive quantity

    def _process_data(self, verbose=False):

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
        
        if verbose:
            print(f"Signal rows : {len(signal_features)}")
            print(f"Signal columns : {len(signal_features[0])}")
            print(signal_features,end='\n\n')

        return prices, signal_features

    def _get_local_state(self):
        return self._quantity

    def _calculate_reward(self, action, terminal=False):
        next_price = self.prices[int(self._current_tick+1)]
        current_price = self.prices[int(self._current_tick)]
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

    def best_action(self, ):
        next_price = self.prices[int(self._current_tick+1)]
        current_price = self.prices[int(self._current_tick)]
        threshold = 0.05
        
        if next_price/current_price < 1-threshold:
            action= 2
        elif next_price/current_price > 1+ threshold:
            action =1
        else:
            action= 0

        return self.step(action=action)