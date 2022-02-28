import gym
from models.policyGradient.policygradient import *
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import *
import matplotlib.pyplot as plt
import pandas as pd

#DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
#RENDER = False  # rendering wastes time

df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_1h.csv", delimiter= ",")

window_size = 400
frame_len = 6
start_index = window_size
end_index = len(df_btc)
#print("ça commence")
#env = gym.make('CartPole-v0')
env = CryptoEnv(df = df_btc , window_size=window_size, frame_len= frame_len)
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        #if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            #if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_

plt.figure(figsize=(16, 6))
env.render_all(window='large')
#plt.savefig("output.png")
plt.show()