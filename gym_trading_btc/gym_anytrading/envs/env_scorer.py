from .bitcoin_env import CryptoEnv


class CryptoEnv_scorer():
    def __init__(self, df, window_size, frame_len) -> None:
        
        self._actual_env = CryptoEnv(df, window_size, frame_len)
        
        self._random_env = CryptoEnv(df, window_size, frame_len)
        self._random_reward_ep = 0
        self._random_profit = 0
        
        self._optimal_env = CryptoEnv(df, window_size, frame_len)
        self._optimal_reward_ep = 0
        self._optimal_profit = 0
        

    def reset(self):
        obs = self._actual_env.reset()
        self._random_env.merge(self._actual_env)
        self._random_env.merge(self._actual_env)
        
        self._random_reward_ep = 0
        self._random_profit = 0
        
        self._optimal_reward_ep = 0
        self._optimal_profit = 0
        
        return obs
    
    def get_observation(self):
        return self._actual_env._get_observation()
    
    def step(self, action):
        
        random_action = self._random_env.action_space.sample()
        observation, reward, done, info = self._random_env.step(action)
        self._random_reward_ep += reward
        if not done:
            self._random_profit += reward
        
        
        return self._actual_env.step(action=action)