
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
        

    def play_episodes(self, num_episodes):
        random_profits = []
        agent_profits = []
        optimal_profits = []
        iters = tqdm(range(num_episodes))
        for i in iters:
            random_profit, agent_profit, optimal_profit = self.play_episode(i)
            random_profits.append(random_profit)
            agent_profits.append(agent_profit)
            optimal_profits.append(optimal_profit)
            if (i % 100 == 0):
                iters.set_description(f"> Episode {i:5}  | random  {random_profit:10.2f}  | agent  {agent_profit:10.2f}  | optimal  {optimal_profit:10.2f}  | score  {100 * float(agent_profit - random_profit) / (optimal_profit - random_profit):10.2f}%")
        
        return random_profits, agent_profits, optimal_profits


    def play_episode(self, index):
        self.env.reset()
        tick = self.env._padding_tick
        random_profit = self.play_agent(self.random, index)
        self.env.reset_to(tick)
        agent_profit = self.play_agent(self.agent, index)
        self.env.reset_to(tick)
        optimal_profit = self.play_optimal()

        return random_profit, agent_profit, optimal_profit

    def play_optimal(self):

        start = self.env._current_tick
        steps = self.env.frame_len
        end = start + steps
        actions = [Actions.Stay] * steps
        prices = self.env.prices[start:end]

        def get_current_profit(prices, actions):
            profit = 0
            count = 0
            for i in range(steps):
                if (actions[i] == Actions.Buy):
                    count += 1
                    profit -= prices[i] # TODO fees
                elif (actions[i] == Actions.Sell):
                    count -= 1
                    profit += prices[i] # TODO fees
            profit += count * prices[steps-1]
            return profit

        profit = get_current_profit(prices, actions)

        for i in range(3 * steps):
            rand_index = rand.randint(0, steps - 1)
            initial_action = actions[rand_index]
            rand_action = initial_action
            while(rand_action == initial_action):
                rand_action = np.random.choice(list(Actions))
            actions[rand_index] = rand_action
            new_profit = get_current_profit(prices, actions)
            if (new_profit <= profit):
                actions[rand_index] = initial_action
            else:
                profit = new_profit
            #print(profit)

        return profit

    def play_agent(self, agent, index):

        state = self.env._get_observation()
        terminal = False

        while not terminal:

            action = agent.predict(state)
            state_after, reward, terminal, _ = self.env.step(action)   
            agent.learn(state, action, state_after, reward, terminal)
            
            state = state_after

        agent.learn_episode(index)

        return reward