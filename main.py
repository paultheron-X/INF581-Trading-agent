import matplotlib.pyplot as plt
import pandas as pd
from gym_trading_btc.envs.bitcoin_env import CryptoEnv
from analytics.env_scorer import CryptoEnvScorer
import warnings

from models.dqn import DQNAgentDeepsense
from models.a2c import A2CAgent

from config_mods import config_dqn_deepsense as config


df_btc = pd.read_csv(config["df_path"], delimiter=",")

env = CryptoEnv(**config)
agent = A2CAgent(**config)
scorer = CryptoEnvScorer(env, agent, **config)

if config['load']:
    agent.load_model(**config)

num_episodes = config['num_episode']

random_profit, agent_profit, optimal_profit = scorer.train_episodes(num_episodes)

if config['save']:
    agent.save_model(**config)

def plot_asolute(random_profit, agent_profit, optimal_profit, title='figs/asolute-agent-profit.png'):
    plt.plot(range(num_episodes), random_profit, label="Random profit")
    plt.plot(range(num_episodes), agent_profit, label="Agent profit")
    plt.plot(range(num_episodes), optimal_profit, label="'Optimal' profit")
    plt.legend()
    plt.savefig(title)
    plt.clf()

def plot_relative(random_profit, agent_profit, optimal_profit, title='figs/relative-agent-profit.png'):
    relative = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(num_episodes):
            r = random_profit[i]
            a = agent_profit[i]
            o = optimal_profit[i]
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