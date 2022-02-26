import os, sys
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

from .constants import *
from gym_trading_btc.gym_anytrading.envs import *

df_btc = pd.read_csv("gym_trading_btc/gym_anytrading/datasets/data/Bitstamp_BTCUSD_2017-2022_minute.csv", delimiter= ",")

class DQNSolver(nn.Module):
    def __init__(self, input_size = WINDOW_SIZE*NUM_COLUMNS, n_actions = NUM_ACTIONS) -> None:
        super(DQNSolver, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(input_size, 256),
                #nn.Dropout(dropout),
                nn.ReLU(),
                #('hidden' , nn.Linear(hidden_size, hidden_size)),
                nn.Dropout(p = 0.1),
                #('relu2' , nn.ReLU()),
                nn.Linear(256, n_actions),
                nn.Softmax()
            )
        
    def forward(self, x):
        return self.fc(x)

class DQNAgent: 
    def __init__(self, state_space = WINDOW_SIZE*NUM_COLUMNS, action_space = NUM_ACTIONS, dropout = 0.2, hidden_size = 128,  pretrained = False, lr = 0.00025, gamma=0.9, max_mem_size = 30000, exploration_rate = 1.0, exploration_decay = 0.99, exploration_min = 0.1,  batch_size = 32) -> None:
         self.state_space = state_space
         self.action_space = action_space
         self.pretrained = pretrained
         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
         
         self.dqn_validation = DQNSolver(input_size =state_space, 
                              n_actions=action_space,
                              ).to(self.device)
         
         self.dqn_target = DQNSolver(input_size =state_space, 
                              n_actions=action_space,
                              ).to(self.device)
         
         self.lr = lr
         
         self.optimizer = torch.optim.RMSprop(self.dqn_validation.parameters(), lr = self.lr)
         self.loss = nn.HuberLoss().to(self.device)
         self.gamma = gamma
         
         self.memory_size = max_mem_size
         self.exploration_rate = exploration_rate      # To preserve from getting stuck
         self.exploration_decay = exploration_decay
         self.exploration_min = exploration_min
         
         self.memory = deque(maxlen = max_mem_size)
         
         self.current_position = 0
         self.is_full = 0
         
         self.batch_size = batch_size
         
         self.env = CryptoEnv(df=df_btc, window_size= WINDOW_SIZE, frame_len = FRAME_LEN)
         
    def predict(self, state):
        #state = state[np.newaxis, ...]
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
        self.memory.append((state, action, reward, issue, terminal))
    
    def compute_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        return batch

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
        if self.current_position > self.batch_size:
            state ,action ,reward ,issue ,term = self.compute_batch()
            
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