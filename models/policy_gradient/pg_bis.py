import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas as pd
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import *

ENV_NAME = "CartPole-v1"

# Since the goal is to attain an average return of 195, horizon should be larger than 195 steps (say 300 for instance)
EPISODE_DURATION = 100

ALPHA_INIT = 0.1
SCORE = 100000000.0
NUM_EPISODES = 100
LEFT = 0
RIGHT = 1

VERBOSE = True

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def draw_action(s, theta):
    prob_push_right = logistic_regression(s, theta)

    r = np.random.rand()
    if r < prob_push_right:
        return 1
    else:
        return 0

def logistic_regression(s, theta):
    prob_push_right = sigmoid(np.dot(s, np.transpose(theta)))
    return prob_push_right

# Generate an episode
def play_one_episode(env, theta, max_episode_length=EPISODE_DURATION, render=False):
    s_t = env.reset()

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    for t in range(max_episode_length):

        a_t = draw_action(s_t, theta)
        s_t, r_t, done, info = env.step(a_t)

        episode_states.append(s_t)
        episode_actions.append(a_t)
        episode_rewards.append(r_t)

        if done:
            break

    return episode_states, episode_actions, episode_rewards

def score_on_multiple_episodes(env, theta, score=SCORE, num_episodes=NUM_EPISODES, max_episode_length=EPISODE_DURATION, render=False):
        
    num_success = 0
    average_return = 0
    num_consecutive_success = [0]

    for episode_index in range(num_episodes):
        _, _, episode_rewards = play_one_episode(env, theta, max_episode_length, render)

        total_rewards = sum(episode_rewards)

        if total_rewards >= score:
            num_success += 1
            num_consecutive_success[-1] += 1
        else:
            num_consecutive_success.append(0)

        average_return += (1.0 / num_episodes) * total_rewards

        if render:
            print("Test Episode {0}: Total Reward = {1} - Success = {2}".format(episode_index,total_rewards,total_rewards>score))

    if max(num_consecutive_success) >= 15:    # MAY BE ADAPTED TO SPEED UP THE LERNING PROCEDURE
        success = True
    else:
        success = False
        
    return success, num_success, average_return

# Returns Policy Gradient for a given episode
def compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta):

    H = len(episode_rewards)
    PG = 0

    for t in range(H):

        prob_push_right = logistic_regression(episode_states[t], theta)
        a_t = episode_actions[t]
        R_t = sum(episode_rewards[t::])
        if a_t == LEFT:
            g_theta_log_pi = - prob_push_right * episode_states[t] * R_t
        else:
            prob_push_left = (1 - prob_push_right)
            g_theta_log_pi = prob_push_left * episode_states[t] * R_t

        PG += g_theta_log_pi

    return PG

# Train the agent got an average reward greater or equals to 195 over 100 consecutive trials
def train(env, theta_init, max_episode_length = EPISODE_DURATION, alpha_init = ALPHA_INIT):

    theta = theta_init
    episode_index = 0
    average_returns = []

    success, _, R = score_on_multiple_episodes(env, theta)
    average_returns.append(R)

    # Train until success
    while (not success):

        # Rollout
        episode_states, episode_actions, episode_rewards = play_one_episode(env, theta, max_episode_length)

        # Schedule step size
        #alpha = alpha_init
        alpha = alpha_init / (1 + episode_index)

        # Compute gradient
        PG = compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta)

        # Do gradient ascent
        theta += alpha * PG

        # Test new policy
        success, _, R = score_on_multiple_episodes(env, theta, render=False)

        # Monitoring
        average_returns.append(R)

        episode_index += 1

        if VERBOSE:
            print("Episode {0}, average return: {1}".format(episode_index, R))

    return theta, episode_index, average_returns

#np.random.seed(1234)
df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_1h.csv", delimiter= ",")

window_size = 400
frame_len = 6
start_index = window_size
end_index = len(df_btc)

env = CryptoEnv(df = df_btc , window_size=window_size, frame_len= frame_len)
#env = gym.make(ENV_NAME)
#RenderWrapper.register(env, force_gif=True)
#env.seed(1234)

dim = env.observation_space.shape[0]

# Init parameters to random
theta_init = np.random.randn(1, dim)

# Train the agent
theta, i, average_returns = train(env, theta_init)

print("Solved after {} iterations".format(i))

score_on_multiple_episodes(env, theta, num_episodes=10, render=True)
#env.render_wrapper.make_gif("ex4")
print("fin num 1")
# Show training curve
plt.plot(range(len(average_returns)),average_returns)
plt.title("Average reward on 100 episodes")
plt.xlabel("Training Steps")
plt.ylabel("Reward")
print("fin num 2")
plt.show()
print("c'est la fin")
env.close()