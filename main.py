import gym 
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import *
import gym_trading_btc.gym_anytrading as gym_anytrading
import pandas as pd
import matplotlib.pyplot as plt

df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_1h.csv", delimiter= ",")


window_size = 10
frame_len = 200
start_index = window_size
end_index = len(df_btc)

env = CryptoEnv(df = df_btc , window_size=window_size, frame_len= frame_len)

observation = env.reset()

while True:
    observation = observation[np.newaxis, ...]

    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    
    if done:
        print("info:", info)
        break

plt.figure(figsize=(16, 6))
env.render_all()
#plt.savefig("output.png")
plt.show()