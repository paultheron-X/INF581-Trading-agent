import numpy as np
from torch import threshold
import matplotlib.pyplot as plt
from gym import spaces
from gym.utils import seeding
from enum import Enum

from config_mods import *

class Actions(Enum):
    Sell = 2
    Stay = 0
    Buy = 1

class CryptoEnv:
    
    metadata = {'render.modes': ['human']}
       
    def __init__(self, df, window_size, frame_len, name = 'base'):
        assert df.ndim == 2
        assert df.shape[0] > window_size
        
        
        self.trade_fee_bid_percent = 0.01  # unit
        self._unit = 1  # units of btc
        self._quantity = 0  # positive quantity
        
        self.name = name
        
        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.frame_len = min(frame_len, len(self.prices) - window_size)
        #self.shape = (window_size, self.signal_features.shape[1])
        self.shape = (window_size*7, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)


        # episode
        self._max_start_tick = len(self.prices) - self.frame_len
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 2
        #self._start_budget = start_budget
        self._done = None
        self._current_tick = None
        self._padding_tick = None
        #self._last_trade_tick = None
        self._position_history = None
        self._total_reward = None
        self._last_reward = None
        #self._budget = None
        #self._quantity = None
        #self._total_profit = None
        #self._first_rendering = None
        self.history = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self): #, history = History(logger, config), replay_memory = ReplayMemory(logger, config) ):
        self._done = False
        self._padding_tick = int(np.floor(np.random.rand() * self._max_start_tick))
        self._current_tick = self._start_tick + self._padding_tick
        self._end_tick = self._current_tick + self.frame_len
        #self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._last_reward = 0.
        #self._quantity = 0.
        self._position_history = [Actions.Stay] * self._start_tick
        self._total_profit = 0  # unit
        #self._budget = self._start_budget
        #self._first_rendering = True
        self.history = {}

        return self._get_observation()
    
    def merge(self, model_env):
        self._done = model_env._done
        self._padding_tick = model_env._padding_tick
        self._current_tick = model_env._current_tick
        self._end_tick = model_env._end_tick
        
        self._total_reward = model_env._end_tick
        self._last_reward = model_env._last_reward
       
        self._position_history = model_env._position_history
        self._total_profit = model_env._total_profit 
        
        self.history = model_env.history

    def step(self, action):
        self._current_tick += 1
        if self._current_tick == self._end_tick:
            self._done = True
            # Il faut tout revendre pour tomber à zero action short ou possédée
            step_reward = self._update_profit_reward(action=action, terminal=True)
            #print(" > For this last step, Action :  " + str(action) + " | Reward : " +
             #     str(step_reward) + " | Total profit " + str(self._total_profit))
        else:
            self._done = False
            step_reward = self._update_profit_reward(action)
           # print("     > For this step, Action :  " + str(action) + " | Reward : " +
            #      str(step_reward) + " | Total profit " + str(self._total_profit))

        self._last_reward = step_reward
        self._position_history.append(action)
        observation = self._get_observation()

        info = dict(
            current_reward=self._last_reward,
            total_profit=self._total_reward,
            position=action
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        """Return the current state :
        - short and long quantity
        - "open", "high", "low", "close", "Volume BTC", "Volume USD" for the last 'window-size' period

        Returns:
            tuple of 2 things :
            - (short quantity, long quantity) at first argument
            - array of array : other information for each periode, one periode par row
        """
        #return np.concatenate((np.array(self._get_local_state()), self.signal_features[int(self._current_tick-self.window_size):int(self._current_tick)]), axis=None)
        return np.ravel(self.signal_features[int(self._current_tick-self.window_size):int(self._current_tick)])
    
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Actions.Sell:
                color = 'red'
            elif position == Actions.Buy:
                color = 'green'
            elif position == Actions.Stay:
                color = 'yellow'
            if color:
                plt.bar(tick, 1, color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, mode='human', window='local'):
        if window == 'local':
            start = 0
            prices = np.array(
                self.prices[
                    self._padding_tick:
                    int(self._current_tick+1)
                ]
            )
        elif window == 'large':
            start = self._padding_tick
            prices = np.array(self.prices)
        else:
            raise NotImplementedError

        plt.plot(prices)
        position_history = np.array(self._position_history)
        buy_ind = np.array(
            [int(start+i) for i, a in enumerate(position_history) if a == Actions.Buy.value])
        buy_val = np.array([self.prices[a] for a in buy_ind])
        plt.scatter(buy_ind, buy_val, color='green')

        sell_ind = np.array(
            [int(start+i) for i, a in enumerate(position_history) if a == Actions.Sell.value])
        sell_val = np.array([self.prices[a] for a in sell_ind])
        plt.scatter(sell_ind, sell_val, color='red')

        stay_ind = np.array(
            [int(start+i) for i, a in enumerate(position_history) if a == Actions.Stay.value])
        stay_val = np.array([self.prices[a] for a in stay_ind])
        plt.scatter(stay_ind, stay_val, color='yellow')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Portfolio: %.6f" % self._total_profit
        )
        plt.show()

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _get_local_state(self):
        return None

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
            #print('quantity ' + self.name, self._quantity)
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

    def best_action(self):
        """Function to use in order to have the best possible action at a time t

        Returns:
            tuple: same return than step function fot this precise 'best' action
        """
        
        next_price = self.prices[int(self._current_tick+1)]
        current_price = self.prices[int(self._current_tick)]
        threshold = 0
        
        
        if (next_price/current_price) < (1  - threshold):  
            action = 2
        elif (next_price/current_price) > 1 + threshold:
            action = 1
        else:
            action= 0

        return self.step(action=action)
    
    
    
    