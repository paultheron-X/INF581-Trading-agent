import numpy as np

from .crypto_trading_env import CryptoTradingEnv
from .trading_env import Actions, Positions


class CryptoEnv(CryptoTradingEnv):

    def __init__(self, df, window_size, frame_len, start_budget=100000):
        super().__init__(df, window_size, frame_len, start_budget)
        self.trade_fee_bid_percent = 0.01  # unit
        self._unit = 1 #units of btc
        

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

    def _calculate_reward(self, action, terminal = False):
        next_price = self.prices[int(self._current_tick+1)]
        current_price = self.prices[int(self._current_tick)]
        
        if action == 1: # Buy
            self._long +=1
            current_transaction_amount = self._unit * current_price * (1+self.trade_fee_bid_percent) 
            self._total_profit += current_transaction_amount
        elif action == 2: #Sell
            self._short +=1
            current_transaction_amount = - self._unit * current_price * (1+self.trade_fee_bid_percent)
            self._total_profit += current_transaction_amount
        
        reward = (self._long - self._short) * self._unit * (next_price - current_price)   # Quid du trade fee bid percent ?

        
        if terminal:
            # etat terminal -> on revend tout au prix du marché pour avoir notre profit
            positive_transation_amount = -self._long * self._unit * current_price * (1+self.trade_fee_bid_percent)   # Sell everything I own
            negative_transation_amount = self._short * self._unit * current_price * (1+self.trade_fee_bid_percent)   # Buy everything I short
            self._total_profit += positive_transation_amount + negative_transation_amount
        
        return reward

    def _update_profit_reward(self, action, terminal = False): # Je mets dans CryptoTrading la MaJ du buget et de la quantité
        if not terminal:
            instant_reward = self._calculate_reward(action = action)
            self._total_reward +=instant_reward
            return instant_reward
        else:                                               # l'etat est terminal, il faut en plus vendre tout ce qu'on a
            instant_reward = self._calculate_reward(action = action)
            self._total_reward +=instant_reward
            return instant_reward

    def max_possible_profit(self):
        # la fonction est à réécrire, mais dans l'idée, c'est ça
        # sachant qu'on ne prend pas en compte les fees :
        # il faudrait compter en benef les moindres augmentations entre 2 temps
        start_tick = self._start_tick + self._padding_tick
        last_trade_tick = start_tick - 1
        profit = 0.
        for i in range(start_tick, self._end_tick + 1):
            for j in range(i + 1, self._end_tick + 1):
                d = self.prices[j] / self.prices[i]
                if d > profit:
                    profit = d
        return profit
