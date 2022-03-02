import matplotlib.pyplot as plt
import pandas as pd
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import CryptoEnv
from analytics.env_scorer import CryptoEnvScorer

from models.dqn import DQNAgentDeepsense
from models.policyGradient import A2CAgent

from config_mods import config_dqn_deepsense as config


df_btc = pd.read_csv(config["df_path"], delimiter=",")

env = CryptoEnv(**config)
agent = A2CAgent(**config)
scorer = CryptoEnvScorer(env, agent, **config)

num_episodes = config['num_episode']

random_profit, agent_profit, optimal_profit = scorer.play_episodes(num_episodes)

def plot_asolute(random_profit, agent_profit, optimal_profit, title='figs/asolute-agent-profit.png'):
    plt.plot(range(num_episodes), random_profit, label="Random profit")
    plt.plot(range(num_episodes), agent_profit, label="Agent profit")
    plt.plot(range(num_episodes), optimal_profit, label="'Optimal' profit")
    plt.legend()
    plt.savefig(title)
    plt.clf()

def plot_relative(random_profit, agent_profit, optimal_profit, title='figs/relative-agent-profit.png'):
    relative = []
    for r,a,o in zip(random_profit, agent_profit, optimal_profit):
        try:
            relative.append(float(a - r) / (o - r))
        except:
            relative.append(0)
    plt.plot(range(num_episodes), relative, label="Relative profit for agent")
    plt.axhline(y = 1, linestyle = ':', label = "Optimal")
    plt.axhline(y = 0, linestyle = ':', label = "Random")
    plt.legend()
    plt.savefig(title)


plot_asolute(random_profit, agent_profit, optimal_profit)
plot_relative(random_profit, agent_profit, optimal_profit)