import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import pickle
import json
import os
import copy 

from models import Agent


class PolicyNet(nn.Module):
    def __init__(self, **kwargs):
        super(PolicyNet, self).__init__()
        
        self.hidden_size_mlp = kwargs["hidden_size"]
        self.input_size = kwargs["window_size"] * kwargs['num_features']
        self.num_features = kwargs["num_features"]
        
        self.policy = torch.nn.Sequential()
        
        block_input = torch.nn.Sequential(
                nn.Linear(in_features = self.input_size, 
                          out_features =self.hidden_size_mlp[0]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["linear_dropout"])
            )
        self.policy.add_module('actor_block_input', copy.copy(block_input))
        
        for ind in range(1, len(self.hidden_size_mlp)):
            block = torch.nn.Sequential(
                nn.Linear(in_features = self.hidden_size_mlp[ind-1], 
                          out_features = self.hidden_size_mlp[ind]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["linear_dropout"])
            )
            self.policy.add_module('actor_block_'+ str(ind), copy.copy(block))
        
        block_fin = torch.nn.Sequential(
                nn.Linear(kwargs["hidden_size"][len(self.hidden_size_mlp)-1], kwargs["num_actions"]),
                nn.Sigmoid()
            )

        self.policy.add_module('actor_block_output', copy.copy(block_fin))
        

    def forward(self, x):
        x = self.policy(x)
        return x

class PolicyGradientAgent(Agent):
    #def __init__(self, **config):
    def __init__(self, **config):
        super().__init__(**config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.policy = PolicyNet(**config)
        
        self.optimizer = optim.RMSprop(self.policy.policy.parameters(), lr=config['lr'])
        self.eps = np.finfo(np.float32).eps.item()
        
        self.gamma = config["gamma"]

        
        
        self.states = []     #List to store the states
        self.actions = []    #List to store the actions
        self.rewards = []
        self.rr = 0

    
    #------------- Override the inheritance functions from Agent

    def predict(self,state):
        try:
            state = torch.from_numpy(state).float().unsqueeze(0)
        except TypeError:
            pass
        probs = self.policy(state)
        print(probs)
        m = Categorical(probs)
        action = m.sample()
        return action
    
    def learn(self, previous_state, action, next_state, reward, terminal):
        self.states.append(previous_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.rr +=reward
        return super().learn(previous_state, action, next_state, reward, terminal)
    
    def learn_episode(self, episode_num, **kwargs):
        self.optimizer.zero_grad()
        loss_sum =0
        self.rewards[-1] = -1
        
        self.rewards = self._compute_returns(next_value = None, rewards = self.rewards, masks = None, gamma=0.99)
        rewards_mean = np.mean(self.rewards)
        rewards_std = np.std(self.rewards)
        
        
        for j in range(len(self.rewards)):
            self.rewards[j] = (self.rewards[j] - rewards_mean) / rewards_std
        for j in range(len(self.rewards)):
            inp = torch.from_numpy(self.states[j]).unsqueeze(0)
            inp = inp.type('torch.FloatTensor')
            acts = self.predict(inp)
            m = Categorical(acts)
            print(m)
            print(self.actions[j])
            print(self.rewards[j])
            loss = -m.log_prob(self.actions[j]) * self.rewards[j]
            loss.backward()
            loss_sum+=loss
        self.optimizer.step()
        
        self.states = []     #List to store the states
        self.actions = []    #List to store the actions
        self.rewards = []

    def save_model(self, **kwargs):
        path = kwargs['save_path']
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), path + "/policy.pt")  
        with open(path + "/memory.pkl", "wb") as f:
            pickle.dump(self.memory, f)
        with open(path + "/model_config.txt", "w") as f:
            f.write(json.dumps(kwargs, indent = 6))
        print("> Ending simulation, PolicyGradient model successfully saved")
    
    def load_model(self, **kwargs):
        path = kwargs['load_path']
        self.policy.load_state_dict(torch.load(path + "/policy.pt"))
        file_memory = open(path + "/memory.pkl", 'rb') 
        self.memory = pickle.load(file_memory)
        print("> Starting simulation, PolicyGradient model successfully loaded")

    
    #-------------- Private part
    
    def _compute_returns(self, next_value, rewards, masks, gamma=0.99):
        cum_reward =0
        for j in reversed(range(len(rewards))):
            if rewards[j]!=0:
                cum_reward = cum_reward*gamma + rewards[j]
                rewards[j]=cum_reward
        return rewards


