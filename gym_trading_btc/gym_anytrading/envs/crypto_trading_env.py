import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Actions(Enum):
    Sell = -1
    Stay = 0
    Buy = 1


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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._max_start_tick = len(self.prices) - self.frame_len
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
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

    def reset(self):
        self._done = False
        self._padding_tick = np.floor(np.random.rand() * self._max_start_tick)
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

    def step(self, action):
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True
            # Il faut tout revendre pour tomber à zero action short ou possédée
            step_reward = self._update_profit_reward(
                action=action, terminal=True)
            print(" > For this last step, Action :  " + str(action) + " | Reward : " +
                  str(step_reward) + " | Total profit " + str(self._total_profit))
        else:
            self._done = False
            step_reward = self._update_profit_reward(action)
            print("     > For this step, Action :  " + str(action) + " | Reward : " +
                  str(step_reward) + " | Total profit " + str(self._total_profit))

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
        return self._get_local_state(self), \
            self.signal_features[
                int(self._current_tick-self.window_size):
                int(self._current_tick)
        ]

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
                color = 'yellow'
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
            [int(start+i) for i, a in enumerate(position_history) if a == Actions.Buy])
        buy_val = np.array([self.prices[a] for a in buy_ind])
        plt.scatter(buy_ind, buy_val, color='green')

        sell_ind = np.array(
            [int(start+i) for i, a in enumerate(position_history) if a == Actions.Sell])
        sell_val = np.array([self.prices[a] for a in sell_ind])
        plt.scatter(sell_ind, sell_val, color='red')

        stay_ind = np.array(
            [int(start+i) for i, a in enumerate(position_history) if a == Actions.Stay])
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

    def _process_data(self):
        raise NotImplementedError

    def _get_local_state(self):
        return None

    def _calculate_reward(self, action, terminal=False):
        raise NotImplementedError

    def _update_profit_reward(self, action, terminal=False):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
