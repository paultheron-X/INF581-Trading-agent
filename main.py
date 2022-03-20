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
from models.policy_gradient import PolicyGradientAgent
from models.classifier import ClassifierAgent
from models.utils.initialization import init_weights

from config_mods import *

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--df_name", help="df_name", required=False)
parser.add_argument("--save_path", help="save_path", required=False)
parser.add_argument("--load_path", help="load_path", required=False)
parser.add_argument("--num_episode", help="num_episode", required=False)
parser.add_argument("--save", help="save", required=False)
parser.add_argument("--load", help="load", required=False)
parser.add_argument("--lr", help="lr", required=False)
parser.add_argument("--config", required=False)
parser.add_argument("--classifier_model", required=False)
parser.add_argument("--classifier_objective", required=False)
args = parser.parse_args()

if args.config is not None:
    if args.config == 'dqn_base':
        config = config_dqn_base
        agent = DQNAgentDeepsense(**config)
        agent.dqn_validation.apply(init_weights)
        agent.dqn_target.apply(init_weights)
    elif args.config == "dqn_deepsense":
        config = config_dqn_deepsense
        agent = DQNAgentDeepsense(**config)
        agent.dqn_validation.apply(init_weights)
        agent.dqn_target.apply(init_weights)
    elif args.config == "config_a2c":
        config = config_a2c
        agent = A2CAgent(**config)
        agent.actor_critic.apply(init_weights)
    elif args.config == "classifier":
        config = config_classifier


else:
    config = config_a2c
    agent = A2CAgent(**config)


if args.df_name is not None:
    config['df_name'] = args.df_name
if args.df_name is not None:
    config['save_path'] = args.save_path
if args.df_name is not None:
    config['load_path'] = args.load_path
if args.df_name is not None:
    config['num_episode'] = int(args.num_episode)
if args.save is not None:
    config['save'] = int(args.save)
if args.load is not None:
    config['load'] = int(args.load)
if args.lr is not None:
    config['lr'] = float(args.lr)
if args.classifier_model is not None:
    config['model'] = args.classifier_model
if args.classifier_objective is not None:
    config['objective'] = args.classifier_objective

# Update path
config['df_path'] = 'gym_trading_btc/datasets/data/' + config['df_name']
df_btc = pd.read_csv(config["df_path"], delimiter=",")

env = CryptoEnv(**config)

if args.config == "classifier":
    config["X_train"], config["Y_train"], config["X_test"], config["Y_test"] = env.get_data()
    agent = ClassifierAgent(**config)

scorer = CryptoEnvScorer(env, agent, **config)

if config['load']:
    agent.load_model(**config)

num_episodes = config['num_episode']

random_profit_ep, agent_profit_ep, optimal_profit_ep, random_profit_val, agent_profit_val, optimal_profit_val = scorer.train_episodes(
    num_episodes)

if config['save']:
    agent.save_model(**config)


def plot_asolute(random_profit, agent_profit, optimal_profit, title):
    #plt.plot(range(num_episodes), random_profit, label="Random profit")
    plt.plot(range(len(agent_profit)), agent_profit, label="Agent profit")
    plt.plot(range(len(optimal_profit)),
             optimal_profit, label="'Optimal' profit")
    plt.legend(
        f'Model {args.config} ; DF f{config["df_name"]} \nLR {config["lr"]} ; Pretrained {"True" if config["load"] == 1 else "False"}')
    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    os.makedirs("figs", exist_ok=True)
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
    plt.plot(range(len(agent_profit)), relative,
             label="Relative profit for agent")
    plt.axhline(y=1, linestyle=':', label="Optimal")
    plt.axhline(y=0, linestyle=':', label="Stay")
    plt.legend(
        f'Model {args.config} ; DF f{config["df_name"]} \nLR {config["lr"]} ; Pretrained {"True" if config["load"] == 1 else "False"}')
    os.makedirs("figs", exist_ok=True)

    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    plt.savefig(title+"_"+dt_string+".png")


plot_asolute(random_profit_ep, agent_profit_ep, optimal_profit_ep,
             title='figs/asolute-agent-profit_episode')
plot_relative(random_profit_ep, agent_profit_ep, optimal_profit_ep,
              title='figs/relative-agent-profit_episode')
plot_asolute(random_profit_val, agent_profit_val, optimal_profit_val,
             title='figs/asolute-agent-profit_validation')
plot_relative(random_profit_val, agent_profit_val, optimal_profit_val,
              title='figs/relative-agent-profit_validation')
