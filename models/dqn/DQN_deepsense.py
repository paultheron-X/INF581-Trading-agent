import os, sys
from re import X
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
import numpy as np
import random
import pickle 
from collections import deque
import pandas as pd

from gym_trading_btc.gym_anytrading.envs import *

from config_mods import config_dqn_deepsense as config 

from .replay_memory import Memory

class DQNSolver(nn.Module):
    def __init__(self, input_size, n_actions, dropout, hidden_size, filter_size, kernel_size, dropout_conv) -> None:
        super(DQNSolver, self).__init__()
        self.fc = nn.Sequential()
        self.fc.add_module('input', nn.Linear(input_size, hidden_size[0] ))
        self.fc.add_module('relu', nn.LeakyReLU())
        self.fc.add_module('dropout_0', nn.Dropout(p = dropout))

        for ind in range(1, len(hidden_size)):
            self.fc.add_module('hidden' + str(ind), nn.Linear(hidden_size[ind-1], hidden_size[ind]))
            self.fc.add_module('relu' + str(ind), nn.LeakyReLU())
            self.fc.add_module('dropout_0' + str(ind), nn.Dropout(p = dropout))
        
        self.fc.add_module('output', nn.Linear(hidden_size[-1], n_actions ))
        self.fc.add_module('softmax', nn.Softmax())
        
    def forward(self, x):
        print(x)
        print(x.shape)
        y = torch.reshape(x, (3, 2, 2, 7 ))
        print(y.shape)
        print(y)
        quit()
        return self.fc(x)

class DQNAgent_ds: 
    def __init__(self,  window_size =  config['window_size'],
                        num_features = config['num_features'],
                        action_space = config['num_actions'], 
                        dropout = config['dropout_linear'], 
                        hidden_size = config['hidden_size'],  
                        lr = config['lr'], 
                        gamma=config['gamma'], 
                        max_mem_size = config['max_mem_size'],
                        exploration_rate = config['exploration_rate'], 
                        exploration_decay = config['exploration_decay'],
                        exploration_min = config['exploration_min'], 
                        batch_size = config['batch_size'],
                        frame_len = config['frame_len'],
                        dataset_path = config['df_path'],
                        filter_size = config['filter_sizes'],
                        kernel_size = config['kernel_sizes'],
                        dropout_conv = config['dropout_conv']
                        ) -> None:
         
         
         self.state_space = window_size*num_features
         self.action_space = action_space
         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
         
         self.dqn_validation = DQNSolver(
                            input_size =self.state_space, 
                            n_actions=action_space,
                            dropout=dropout,
                            hidden_size=hidden_size, 
                            filter_size = filter_size,
                            kernel_size = kernel_size,
                            dropout_conv = dropout_conv
                            ).to(self.device)
         
         self.dqn_target = DQNSolver(
                            input_size =self.state_space, 
                            n_actions=action_space,
                            dropout=dropout,
                            hidden_size=hidden_size, 
                            filter_size = filter_size,
                            kernel_size = kernel_size,
                            dropout_conv = dropout_conv
                            ).to(self.device)
         
         self.lr = lr
         
         self.optimizer = torch.optim.RMSprop(self.dqn_validation.parameters(), lr = self.lr)
         self.loss = nn.HuberLoss().to(self.device)
         self.gamma = gamma
         
         self.memory_size = max_mem_size
         self.exploration_rate = exploration_rate      # To preserve from getting stuck
         self.exploration_decay = exploration_decay
         self.exploration_min = exploration_min
         
         self.memory = Memory(max_mem_size)
         
         self.current_position = 0
         self.is_full = 0
         
         self.batch_size = batch_size
         
         df_btc = pd.read_csv(dataset_path, delimiter=",")
         
         self.env = CryptoEnv(df=df_btc, window_size=window_size, frame_len = frame_len)
         
    def predict(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            state = torch.from_numpy(state).float()
            action = self.dqn_validation(state.to(self.device)).argmax().unsqueeze(0).unsqueeze(0).cpu()
            return action.item()  
        
    def act(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, action, done
    
    def remember(self, state, action, reward, issue, terminal):
        self.memory.append(state, action, reward, issue, terminal)
    

    def print_infos(self, action, target, current):
        if (self.current_position % 100 ==0):
            print("\n------------Training on " + self.device + " epoch " + str(self.current_position) + " -------------")
            print("exploration_rate", self.exploration_rate)
            #print("Prediction for this step", action)
            #print("Target for this step", action)
            #print("Current for this step", action)
    
    def get_observation(self):
        return self.env._get_observation()
    
    def reset(self):
        observation = self.env.reset()
        return observation, False
       
    def trading_lessons(self):
        
        # compute a random batch from the memory before and pass it, then retrop
        if self.memory.size > self.batch_size:
            state ,action ,reward ,issue ,term = self.memory.compute_batch(self.batch_size)
            
            self.optimizer.zero_grad()
            
            #Q - learning :  target = r + gam * max_a Q(S', a)
            
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            issue = issue.to(self.device)
            term = term.to(self.device)
            
            pred_next = self.dqn_target(issue)
            pred_eval = self.dqn_validation(issue).argmax(1)
            current_pred = self.dqn_validation(state)
            
            action_max = pred_eval.long()
            
            target = current_pred.clone()
            
            batch_index = torch.from_numpy(np.arange(self.batch_size, dtype=np.int32)).long()
            
            #target = reward + self.gamma*torch.mul(self.dqn(issue).max(1).values.unsqueeze(1) , 1-term)
            
            action = torch.ravel(action).long()
            reward = reward.ravel()
            term = term.ravel()
            
            target[batch_index, action] = reward + self.gamma*pred_next[batch_index, action_max]*(1-term)
            
            #current = self.dqn(state).gather(1, action.long())
            
            loss = self.loss(current_pred, target)
            loss.backward()
            self.optimizer.step()
            
            # Eventually reduce the exploration rate
            
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_rate, self.exploration_min)
    
    def save(self, name):
        with open(name+ "/deque.pkl", "wb") as f:
            pickle.dump(self.memory, f)
        with open(name+ "/ending_position.pkl", "wb") as f:
            pickle.dump(self.current_position, f)
        with open(name+ "/num_in_queue.pkl", "wb") as f:
            pickle.dump(self.is_full, f)
    
    def update_params(self):
        self.dqn_target.load_state_dict(self.dqn_validation.state_dict())

    def get_exploration(self):
        return self.exploration_rate