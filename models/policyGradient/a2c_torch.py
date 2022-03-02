import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from copy import copy



from models import Agent
# hyperparameters
hidden_size = 256
learning_rate = 3e-4

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 3000

class ActorCritic(nn.Module):
    def __init__(self, **kwargs):
        super(ActorCritic, self).__init__()
        
        self.hidden_size_mlp = kwargs["hidden_size"]
        self.input_size = kwargs["window_size"]
        self.num_features = kwargs["num_features"]

        self.critic = torch.nn.Sequential()
        
        block_input = torch.nn.Sequential(
                nn.Linear(in_features = self.input_size, 
                          out_features =self.hidden_size_mlp[0]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["dropout_linear"])
            )
        self.critic.add_module('critic_block_input', copy.copy(block_input))
        
        for ind in range(1, len(self.hidden_size_mlp)):
            block = torch.nn.Sequential(
                nn.Linear(in_features = self.hidden_size_mlp[ind-1], 
                          out_features =self.hidden_size_mlp[ind]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["dropout_linear"])
            )
            self.mlp_block.add_module('critic_block_'+ str(ind), copy.copy(block))
        
        block_fin = torch.nn.Sequential(
                nn.Linear(kwargs["hidden_size"][-1], kwargs["num_actions"]),
            )

        self.critic.add_module('critic_block_output', copy.copy(block_fin))
        
        
        self.actor = torch.nn.Sequential()
        
        block_input = torch.nn.Sequential(
                nn.Linear(in_features = self.input_size, 
                          out_features =self.hidden_size_mlp[0]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["dropout_linear"])
            )
        self.actor.add_module('actor_block_input', copy.copy(block_input))
        
        for ind in range(1, len(self.hidden_size_mlp)):
            block = torch.nn.Sequential(
                nn.Linear(in_features = self.hidden_size_mlp[ind-1], 
                          out_features = self.hidden_size_mlp[ind]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["dropout_linear"])
            )
            self.mlp_block.add_module('actor_block_'+ str(ind), copy.copy(block))
        
        block_fin = torch.nn.Sequential(
                nn.Linear(kwargs["hidden_size"][-1], kwargs["num_actions"]),
                nn.Softmax(dim=1)
            )

        self.actor.add_module('actor_block_output', copy.copy(block_fin))
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value
   
class A2CAgent(Agent):
    def __init__(self, **config):
        super().__init__(**config)
        
        self.actor_critic = ActorCritic(**config)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        self.entropy = 0 
        
        self.lastvalue = None
        self.lastdist = None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Override the previous functions from Agent
    
    def predict(self, state):        
        state = torch.FloatTensor(state).to(self.device)
        dist, value = self.actor_critic(state)
        action = dist.sample()
        self.lastvalue = value
        self.lastdist = dist
        return action

    def learn(self, previous_state, action, next_state, reward, terminal):
        self._trading_lessons(previous_state, action, next_state, reward, terminal)

    def learn_episode(self, episode_num, **kwargs):
        
        next_state = torch.FloatTensor(kwargs['next_state']).to(self.device)
        _, next_value = self.actor_critic(next_state)
        returns = self._compute_returns(next_value, self.rewards, self.masks)
        
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # reset params
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        self.entropy = 0 
    
    def print_infos(self):
        print("A2C agent")
    

    #----- Private part
    
    def _trading_lessons(self, previous_state, action, next_state, reward, terminal):  
        reward = 0.1*reward
        
        log_prob = self.lastdist.log_prob(action)
        self.entropy += self.lastdist.entropy().mean()
        
        self.log_probs.append(log_prob)
        self.values.append(self.lastvalue)
        self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
        self.masks.append(torch.FloatTensor(1 - terminal).unsqueeze(1).to(self.device))
        
    def _compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns