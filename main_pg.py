import gym
from models.policyGradient.policygradient import *
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import *
import matplotlib.pyplot as plt
import pandas as pd

df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_1h.csv", delimiter= ",")

window_size = 400
frame_len = 6
start_index = window_size
end_index = len(df_btc)

env = CryptoEnv(df = df_btc , window_size=window_size, frame_len= frame_len)
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print((env.observation_space.shape[0],))
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
    #print(observation.shape)
    while True and observation.shape == (env.observation_space.shape[0],):
        #and (observation.shape == env.observation_space.shape)
        #print(str(observation.shape))
        #if observation.shape == (env.observation_space.shape[0],):
        action = RL.choose_action(observation)
        
        if observation.shape == env.observation_space.shape:
            action = RL.choose_action(observation)
        else:
            action = env.action_space.sample()
        
        observation_, reward, done, info = env.step(action)
        
        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()
            """
            if i_episode == 2999:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            """
            break
        observation = observation_
        #print(str(observation_.shape))

print("> Ending episode")
plt.figure(figsize=(16, 6))
env.render_all(window='large')
#plt.savefig("output.png")
plt.show()

#Pour exécuter ce fichier, il faut commenter la ligne 30 de crypto_trading_en.py et décommenter la ligne 31