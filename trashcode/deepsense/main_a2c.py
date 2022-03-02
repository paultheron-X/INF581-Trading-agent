import gym 
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import *
import gym_trading_btc.gym_anytrading as gym_anytrading
import pandas as pd
import matplotlib.pyplot as plt

from copy import copy

from ...models.a2c.a2c_torch import A2CAgent



df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_1h.csv", delimiter= ",")


window_size = 2
frame_len = 6
start_index = window_size
end_index = len(df_btc)

env = CryptoEnv(df = df_btc , window_size=window_size, frame_len= frame_len)

agent = A2CAgent(env)

max_episodes = 100
for episode in range(max_episodes):
    log_probs = []
    values = []
    rewards = []
    state, terminal = agent.reset()
    current_reward = 0
    issue = state.copy()
    game_profit = 0

    while not terminal:
        
        action = agent.predict(state)
        observation_, reward, done, info = agent.act(action)
        reward = reward*0.1
        
        td_error = agent.learn(state, reward, observation_) 


while True:
    print(observation)
    observation = observation[np.newaxis, ...]

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    
    if done:
        print("> Ending episode")
        break

