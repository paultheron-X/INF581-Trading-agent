
from models.agent import Agent
from gym_trading_btc.envs.bitcoin_env import Actions
import numpy as np
import random as rand
from tqdm import tqdm

class CryptoEnvScorer():

    def __init__(self, env, agent, **config):
        self.env = env
        self.agent = agent
        self.random = Agent(**config)
        
    def test(self):
        self.env.reset(training = False)
        tick = self.env._padding_tick
        random_profit = self.test_agent(self.random)
        self.env.reset_to(tick, training = False)
        agent_profit = self.test_agent(self.agent)
        self.env.reset_to(tick, training = False)
        optimal_profit = self.test_optimal()

        return random_profit, agent_profit, optimal_profit

    def test_agent(self, agent):

        state = self.env._get_observation()
        terminal = False

        while not terminal:

            action = agent.predict(state)
            state_after, reward, terminal, _ = self.env.step(action)   
            
            state = state_after

        return reward


    def test_optimal(self):
        fees = self.env.trade_fee_bid_percent
        unit = self.env.unit
        start = self.env._current_tick
        steps = self.env.frame_len
        end = start + steps
        prices = self.env.prices[start:end]

        profit = 0
        end_buy_price = prices[steps - 1] * (1 + fees)
        end_sell_price = prices[steps - 1] * (1 - fees)
        
        for i in range(steps - 1):
            sell_price = prices[i] * (1 - fees)
            if sell_price > end_buy_price:
                profit += unit * (sell_price - end_buy_price)
                continue
            buy_price = prices[i] * (1 + fees)
            if buy_price < end_sell_price:
                profit += unit * (end_sell_price - buy_price)
                continue

        return profit


    def train_episodes(self, num_episodes):
        random_profits = np.zeros(num_episodes)
        agent_profits = np.zeros(num_episodes)
        optimal_profits = np.zeros(num_episodes)
        iters = tqdm(range(num_episodes), colour='blue')
        for i in iters:
            random_profit, agent_profit, optimal_profit = self.train_episode(i)
            random_profits[i] = random_profit
            agent_profits[i] =  agent_profit
            optimal_profits[i] = optimal_profit
            if (optimal_profit < agent_profit or optimal_profit < random_profit):
                print("[Warning] : non optimal profit found")
            if (i % 100 == 99):
                r,a,o = self.test()
                iters.set_description(f"> Episode {i:5}  | random  {r:10.2f}  | agent  {a:10.2f}  | optimal  {o:10.2f}  | score  {100 * float(a - r) / (o - r):10.2f}%")

        return random_profits, agent_profits, optimal_profits


    def train_episode(self, index):
        self.env.reset(training = True)
        tick = self.env._padding_tick
        random_profit = self.train_agent(self.random, index)
        self.env.reset_to(tick, training = True)
        agent_profit = self.train_agent(self.agent, index)
        self.env.reset_to(tick, training = True)
        optimal_profit = self.test_optimal()

        return random_profit, agent_profit, optimal_profit

    def train_agent(self, agent, index):

        state = self.env._get_observation()
        terminal = False

        while not terminal:

            action = agent.predict(state)
            state_after, reward, terminal, _ = self.env.step(action)   
            agent.learn(state, action, state_after, reward, terminal)
            
            state = state_after

        agent.learn_episode(index, **{'next_state': state_after})

        return reward