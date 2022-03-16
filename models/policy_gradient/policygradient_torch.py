import sys
from turtle import forward
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
import copy


from models import Agent

class PolicyGradient(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.hidden_size_mlp = kwargs["hidden_size"]
        self.input_size = kwargs["window_size"] * kwargs['num_features']
        self.num_features = kwargs["num_features"]
        
        self.policy_net = torch.nn.Sequential()
        
        block_input = torch.nn.Sequential(
                nn.Linear(in_features = self.input_size, 
                          out_features =self.hidden_size_mlp[0]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["dropout_linear"])
            )
        self.policy_net.add_module('policy_gradient_block_input', copy.copy(block_input))
        
        for ind in range(1, len(self.hidden_size_mlp)):
            block = torch.nn.Sequential(
                nn.Linear(in_features = self.hidden_size_mlp[ind-1], 
                          out_features = self.hidden_size_mlp[ind]),
                nn.LeakyReLU(),
                nn.Dropout(p = kwargs["dropout_linear"])
            )
            self.policy_net.add_module('policy_gradient_block_'+ str(ind), copy.copy(block))
        
        block_fin = torch.nn.Sequential(
                nn.Linear(kwargs["hidden_size"][len(self.hidden_size_mlp)-1], kwargs["num_actions"]),
                nn.Softmax(dim = 1)
            )

        self.policy_net.add_module('policy_gradient_block_output', copy.copy(block_fin))
    
    def forward(self,x):
        return self.policy_net(x)
    

class PolicyGradientAgent(Agent):
    def __init__(self, **config):
        super().__init__(**config)
        
        self.policy_gradient = PolicyGradient(**config)
        
        self.optimizer = torch.optim.RMSprop(self.policy_gradient.parameters(), lr=config['lr'])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def predict(self, state):
        return super().predict(state)
        
    def learn(self, previous_state, action, next_state, reward, terminal):
        return super().learn(previous_state, action, next_state, reward, terminal)
        
    def learn_episode(self, episode_num, **kwargs):
        return super().learn_episode(episode_num, **kwargs)
    
    def print_infos(self):
        print("Policy Gradient agent")
        
    def load_model(self, **kwargs):
        return super().load_model(**kwargs)
    
    def save_model(self, **kwargs):
        return super().save_model(**kwargs)