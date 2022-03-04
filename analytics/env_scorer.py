
from models.agent import Agent
from gym_trading_btc.gym_anytrading.envs.bitcoin_env import Actions
import numpy as np
import random as rand
from tqdm import tqdm

class CryptoEnvScorer():

    def __init__(self, env, agent, **config):
        self.env = env
        self.agent = agent
        self.random = Agent(**config)
        

    def play_episodes(self, num_episodes, training =  False):
        random_profits = []
        agent_profits = []
        optimal_profits = []
        iters = tqdm(range(num_episodes), colour='blue')
        for i in iters:
            random_profit, agent_profit, optimal_profit = self.play_episode(i, _training = training)
            random_profits.append(random_profit)
            agent_profits.append(agent_profit)
            optimal_profits.append(optimal_profit)
            if (i % 100 == 0):
                iters.set_description(f"> Episode {i:5}  | random  {random_profit:10.2f}  | agent  {agent_profit:10.2f}  | optimal  {optimal_profit:10.2f}  | score  {100 * float(agent_profit - random_profit) / (optimal_profit - random_profit):10.2f}%")

        return random_profits, agent_profits, optimal_profits


    def play_episode(self, index, _training = True):
        self.env.reset(training = _training)
        tick = self.env._padding_tick
        random_profit = self.play_agent(self.random, index)
        self.env.reset_to(tick, training = _training)
        agent_profit = self.play_agent(self.agent, index)
        self.env.reset_to(tick, training = _training)
        optimal_profit = self.play_optimal()

        return random_profit, agent_profit, optimal_profit

    def play_optimal(self):
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
            buy_price = prices[i] * (1 + fees)
            sell_price = prices[i] * (1 - fees)
            if sell_price > end_buy_price:
                profit += unit * (sell_price - end_buy_price)
            elif buy_price < end_sell_price:
                profit += unit * (end_sell_price - buy_price)

        return profit

    def play_agent(self, agent, index):

        state = self.env._get_observation()
        terminal = False

        while not terminal:

            action = agent.predict(state)
            state_after, reward, terminal, _ = self.env.step(action)   
            agent.learn(state, action, state_after, reward, terminal)
            
            state = state_after

        agent.learn_episode(index, **{'next_state': state_after})

        return reward