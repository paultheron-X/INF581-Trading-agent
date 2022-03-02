import gym
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from models.a2c.a2c import *
import pandas as pd
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import *
import matplotlib.pyplot as plt

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_1h.csv", delimiter= ",")

window_size = 400
frame_len = 6
start_index = window_size
end_index = len(df_btc)

env = CryptoEnv(df = df_btc , window_size=window_size, frame_len= frame_len)
#env = gym.make('Pendulum-v0')
env.seed(1)  # reproducible
env = env.unwrapped
print(str(env._end_tick))
print(str(len(env.prices)))
N_S = env.observation_space.shape[0]
N_A_BOUND = env.action_space.n

sess = tf.compat.v1.Session()

actor = Actor(sess, n_features=N_S,n_actions=N_A_BOUND, lr=LR_A)
critic = Critic(sess, n_features=N_S, lr=LR_C)

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("graphs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    observation = env.reset()
    t = 0
    ep_rs = []
    while True and observation.shape == (env.observation_space.shape[0],):

        action = actor.choose_action(observation)
        #print(str(env._current_tick))
        observation_, reward, done, info = env.step(action)
        reward = reward*0.1

        td_error = critic.learn(observation, reward, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(observation, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
        #print(str(reward))
        observation = observation_
        t += 1
        ep_rs.append(reward)
        if t > MAX_EP_STEPS or done:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

print("> Ending episode")
plt.figure(figsize=(16, 6))
env.render_all(window='large')
#plt.savefig("output.png")
plt.show()