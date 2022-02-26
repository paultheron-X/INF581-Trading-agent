import time
from os.path import join

import tensorflow as tf

from config_mods import *

from models.deepsense import *

df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_2017-2022_minute.csv", delimiter= ",")

window_size = 2
frame_len = 6
start_index = window_size
end_index = len(df_btc)

env = CryptoEnv(df = df_btc , window_size=window_size, frame_len = frame_len)

with tf.compat.v1.Session() as sess:
    agent = Agent(sess, logger, config, env)
    print('init')
    agent.train()
    agent.summary_writer.close()


