import matplotlib.pyplot as plt
import pandas as pd
from gym_trading_btc.envs.bitcoin_env import CryptoEnv
from analytics.env_scorer import CryptoEnvScorer
import warnings
import argparse

from models.dqn import DQNAgentDeepsense
from models.a2c import A2CAgent

from config_mods import config_dqn_deepsense as config

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--DF_NAME", help="DF_NAME", required=False)
parser.add_argument("--SAVE_PATH", help="SAVE_PATH", required=False)
parser.add_argument("--LOAD_PATH", help="LOAD_PATH", required=False)
parser.add_argument("--NUM_EPISODE", help="NUM_EPISODE", required=False)
args = parser.parse_args()

if args.df_name is not None:
    config['DF_NAME'] = args.DF_NAME
if args.df_name is not None:
    config['SAVE_PATH'] = args.SAVE_PATH
if args.df_name is not None:
    config['LOAD_PATH'] = args.LOAD_PATH
if args.df_name is not None:
    config['NUM_EPISODE'] = int(args.NUM_EPISODE)


df_btc = pd.read_csv(config["df_path"], delimiter=",")

env = CryptoEnv(**config)
agent = DQNAgentDeepsense(**config)
scorer = CryptoEnvScorer(env, agent, **config)

if config['load']:
    agent.load_model(**config)

num_episodes = config['num_episode']

random_profit_ep, agent_profit_ep, optimal_profit_ep, random_profit_val, agent_profit_val, optimal_profit_val = scorer.train_episodes(num_episodes)

if config['save']:
    agent.save_model(**config)

def plot_asolute(random_profit, agent_profit, optimal_profit, title):
    #plt.plot(range(num_episodes), random_profit, label="Random profit")
    plt.plot(range(len(agent_profit)), agent_profit, label="Agent profit")
    plt.plot(range(len(optimal_profit)), optimal_profit, label="'Optimal' profit")
    plt.legend()
    plt.savefig(title)
    plt.clf()

def plot_relative(random_profit, agent_profit, optimal_profit, title):
    relative = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(len(agent_profit)):
            #r = random_profit[i]
            r = 0
            a = agent_profit[i]
            o = optimal_profit[i]
            try:
                relative.append(float(a - r) / (o - r))
            except:
                relative.append(0)
    plt.plot(range(len(agent_profit)), relative, label="Relative profit for agent")
    plt.axhline(y = 1, linestyle = ':', label = "Optimal")
    plt.axhline(y = 0, linestyle = ':', label = "Stay")
    plt.legend()
    plt.savefig(title)


plot_asolute(random_profit_ep, agent_profit_ep, optimal_profit_ep, title='figs/asolute-agent-profit_episode.png')
plot_relative(random_profit_ep, agent_profit_ep, optimal_profit_ep, title='figs/relative-agent-profit_episode.png')
plot_asolute(random_profit_val, agent_profit_val, optimal_profit_val, title='figs/asolute-agent-profit_validation.png')
plot_relative(random_profit_val, agent_profit_val, optimal_profit_val, title='figs/relative-agent-profit_validation.png')