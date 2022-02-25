import gym 
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import *
import gym_trading_btc.gym_anytrading as gym_anytrading
from gym_trading_btc.gym_anytrading.solvers.qLearning import *
import pandas as pd
import matplotlib.pyplot as plt

#from models.deepsense import *

df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_1h.csv", delimiter= ",")


window_size = 2
frame_len = 6
start_index = window_size
end_index = len(df_btc)

env = CryptoEnv(df = df_btc , window_size=window_size, frame_len= frame_len)

solver = qLearning(env.action_space)

observation = env.reset()

while True:
    observation = observation[np.newaxis, ...]

    action = env.action_space.sample()
    # action = solver(observation)
    observation, reward, done, info = env.step(action)
    
    
    if done:
        print("> Ending episode")
        break

plt.figure(figsize=(16, 6))
env.render_all(window='large')
#plt.savefig("output.png")
plt.show()