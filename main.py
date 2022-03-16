import matplotlib.pyplot as plt
import pandas as pd
from gym_trading_btc.envs.bitcoin_env import CryptoEnv
from analytics.env_scorer import CryptoEnvScorer
import warnings
import datetime
import os
import argparse

from models.dqn import DQNAgentDeepsense
from models.a2c import A2CAgent

from config_mods import config_dqn_deepsense as config

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--df_name", help="df_name", required=False)
parser.add_argument("--save_path", help="save_path", required=False)
parser.add_argument("--load_path", help="load_path", required=False)
parser.add_argument("--num_episode", help="num_episode", required=False)
args = parser.parse_args()

if args.df_name is not None:
    config['df_name'] = args.DF_NAME
if args.df_name is not None:
    config['save_path'] = args.SAVE_PATH
if args.df_name is not None:
    config['load_path'] = args.LOAD_PATH
if args.df_name is not None:
    config['num_episode'] = int(args.NUM_EPISODE)


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
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    os.makedirs("figs", exist_ok = True)
    plt.savefig(title+"_"+dt_string+".png")
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
    os.makedirs("figs", exist_ok = True)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    plt.savefig(title+"_"+dt_string+".png")


plot_asolute(random_profit_ep, agent_profit_ep, optimal_profit_ep, title='figs/asolute-agent-profit_episode')
plot_relative(random_profit_ep, agent_profit_ep, optimal_profit_ep, title='figs/relative-agent-profit_episode')
plot_asolute(random_profit_val, agent_profit_val, optimal_profit_val, title='figs/asolute-agent-profit_validation')
plot_relative(random_profit_val, agent_profit_val, optimal_profit_val, title='figs/relative-agent-profit_validation')