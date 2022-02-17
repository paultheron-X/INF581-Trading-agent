import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from sympy import solve_univariate_inequality

class Actions(Enum):
    Sell = 2
    STAY = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class CryptoTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_len, start_budget):
        assert df.ndim == 2
        assert df.shape[0] > window_size

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.frame_len = min(frame_len, len(self.prices) - window_size)
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._max_start_tick = len(self.prices) - self.frame_len
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._start_budget = start_budget
        self._done = None
        self._current_tick = None
        self._padding_tick = None
        self._last_trade_tick = None
        self._position_history = None
        self._total_reward = None
        self._budget = None
        self._quantity = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._padding_tick = np.floor(np.random.rand() * self._max_start_tick)
        self._current_tick = self._start_tick + self._padding_tick
        self._end_tick = self._current_tick + self.frame_len
        self._last_trade_tick = self._current_tick - 1
        self._total_reward = 0.
        self._quantity = 0.
        self._position_history = (self.window_size * [0])
        self._total_profit = self._start_budget  # unit
        self._budget = self._start_budget
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = action !=0

        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(action)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=action
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[int(self._current_tick-self.window_size):int(self._current_tick)]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position < 0:
                position *= -1
                color = 'red'
            elif position > 0:
                color = 'green'
            elif position == 0:
                color = 'blue'
            if color:
                plt.bar(tick, position, color=color)

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

    def render_all(self, mode='human', window = 'local'):
        if window == 'large':
            window_ticks = np.arange(len(self._position_history))
            plt.plot(self.prices)
            
            self._position_history = np.array(self._position_history)
            
            start = self._current_tick - self.frame_len - self.window_size
            
            buy_ind = np.array([int(start+i) for i,a in enumerate(self._position_history) if a == 1])
            buy_val = np.array([self.prices[a] for a in buy_ind])
            plt.scatter(buy_ind, buy_val, color='green')

            sell_ind = np.array([int(start+i) for i,a in enumerate(self._position_history) if a == 2])
            sell_val = np.array([self.prices[a] for a in sell_ind])
            plt.scatter(sell_ind, sell_val, color='red')

            stay_ind = np.array([int(start+i) for i,a in enumerate(self._position_history) if a == 0])
            stay_val = np.array([self.prices[a] for a in stay_ind])
            plt.scatter(stay_ind, stay_val, color='yellow')
            
            print(stay_ind)

            plt.suptitle(
                "Total Reward: %.6f" % self._total_reward + ' ~ ' +
                "Total Portfolio: %.6f" % self._total_profit
            )
            plt.show()
            
        elif window == 'local':
            window_ticks = np.arange(len(self._position_history))
            
            self.prices_reduced = np.array(self.prices[int(self._current_tick-self.frame_len-self.window_size):int(self._current_tick)])
            
            plt.plot(self.prices_reduced)
            self._position_history = np.array(self._position_history)
            
            
            buy_ind = np.array([int(i) for i,a in enumerate(self._position_history) if a == 1])
            buy_val = np.array([self.prices_reduced[a] for a in buy_ind])
            
            plt.scatter(buy_ind, buy_val, color='green')

            sell_ind = np.array([int(i) for i,a in enumerate(self._position_history) if a == 2])
            sell_val = np.array([self.prices_reduced[a] for a in sell_ind])
            plt.scatter(sell_ind, sell_val, color='red')

            stay_ind = np.array([int(i) for i,a in enumerate(self._position_history) if a == 0])
            stay_val = np.array([self.prices_reduced[a] for a in stay_ind])
            plt.scatter(stay_ind, stay_val, color='yellow')
            
            print(stay_ind)

            plt.suptitle(
                "Total Reward: %.6f" % self._total_reward + ' ~ ' +
                "Total Profit: %.6f" % self._total_profit
            )
            plt.show()
        else:
            raise NotImplementedError

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
