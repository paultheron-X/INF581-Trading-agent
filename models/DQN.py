import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pygame
import torch
import torch.nn as nn
import numpy as np
from game_utils.car import * 
from game_utils.constants import *
import random
import pickle 
from collections import *


class DQNSolver(nn.Module):
    def __init__(self, input_size = NUM_RAYS+1, n_actions = NUM_ACTIONS , dropout = 0.2, hidden_size = 256) -> None:
        super(DQNSolver, self).__init__()
        self.fc = nn.Sequential(
            OrderedDict([
                ('input' , nn.Linear(input_size, hidden_size)),
                #nn.Dropout(dropout),
                ('relu1' , nn.ReLU()),
                #('hidden' , nn.Linear(hidden_size, hidden_size)),
                #nn.Dropout(dropout),
                #('relu2' , nn.ReLU()),
                ('output' , nn.Linear(hidden_size, n_actions)),
                ('sigm' , nn.Softmax())
                ]
            )
        )
        
    def forward(self, x):
        return self.fc(x)

class DQNAgent: 
    def __init__(self, state_space = NUM_RAYS+1, action_space = NUM_ACTIONS, dropout = 0.2, hidden_size = 128,  pretrained = False, lr = 0.00025, gamma=0.9, max_mem_size = 30000, exploration_rate = 1.0, exploration_decay = 0.99, exploration_min = 0.1,  batch_size = 32) -> None:
         self.state_space = state_space
         self.action_space = action_space
         self.pretrained = pretrained
         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
         
         self.dqn_validation = DQNSolver(input_size =state_space, 
                              n_actions=action_space,
                              dropout=dropout,
                              hidden_size=hidden_size
                              ).to(self.device)
         
         self.dqn_target = DQNSolver(input_size =state_space, 
                              n_actions=action_space,
                              dropout=dropout,
                              hidden_size=hidden_size
                              ).to(self.device)
         
         self.lr = lr
         
         self.optimizer = torch.optim.Adam(self.dqn_validation.parameters(), lr = self.lr)
         self.loss = nn.HingeEmbeddingLoss().to(self.device)
         self.gamma = gamma
         
         self.memory_size = max_mem_size
         self.exploration_rate = exploration_rate      # To preserve from getting stuck
         self.exploration_decay = exploration_decay
         self.exploration_min = exploration_min
         
         self.rem_states = torch.zeros(max_mem_size, state_space)
         self.rem_actions = torch.zeros(max_mem_size, 1)
         self.rem_rewards = torch.zeros(max_mem_size, 1)
         self.rem_issues = torch.zeros(max_mem_size, state_space)
         self.rem_terminals = torch.zeros(max_mem_size, 1)
         
         self.current_position = 0
         self.is_full = 0
         
         self.batch_size = batch_size
         
    def act(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            state = torch.from_numpy(state).float()
            action = self.dqn_validation(state.to(self.device)).argmax().unsqueeze(0).unsqueeze(0).cpu()
            return action.item()  
        
    def remember(self, state, action, reward, issue, terminal):
        self.rem_states[self.current_position] =  torch.from_numpy(state).float()
        self.rem_actions[self.current_position] =  torch.tensor(action).float()
        self.rem_rewards[self.current_position] = torch.tensor(reward).float()
        self.rem_issues[self.current_position] = torch.from_numpy(issue).float()
        self.rem_terminals[self.current_position] = torch.tensor(terminal).float()
        
        if True in torch.isinf(torch.from_numpy(state).float()):
            print("tensor", torch.from_numpy(state).float())
            print("np", state)
            pygame.quit()
        if True in torch.isinf(torch.from_numpy(issue).float()):
            print("tensor", torch.from_numpy(issue).float())
            print("np", issue)
            pygame.quit()
        
        self.current_position = (self.current_position + 1) % self.memory_size
        self.is_full = min(self.is_full +1, self.memory_size)
    
    def compute_batch(self):
        #indices = random.choices(range(self.is_full), k = self.batch_size, weights= np.random.exponential(2, self.is_full) )
        #indices = [self.is_full -1 - a for a in (np.random.exponential(2, self.is_full))]
        exp = np.random.exponential(self.batch_size, self.batch_size)
        indices = [(self.is_full - int(a)) % self.memory_size for a in exp]
        
        state_batch = self.rem_states[indices]
        action_batch = self.rem_actions[indices]
        reward_batch = self.rem_rewards[indices]
        issue_batch = self.rem_issues[indices]
        terminal_batch = self.rem_terminals[indices]
        
        return state_batch,action_batch,reward_batch,issue_batch, terminal_batch

    def print_infos(self, action, target, current):
        if PRINT_INFOS and (self.current_position % 100 ==0):
            print("\n------------Training on " + self.device + " epoch " + str(self.current_position) + " -------------")
            print("exploration_rate", self.exploration_rate)
            #print("Prediction for this step", action)
            #print("Target for this step", action)
            #print("Current for this step", action)

       
    def driving_lessons(self):
        
        # compute a random batch from the memory before and pass it, then retrop
        if self.current_position > self.batch_size:
            state,action,reward,issue,term = self.compute_batch()
            
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
            #print(term)
            '''print("action", action)
            print("batch_index", batch_index)
            print(target)
            print(action_max)
            print(target[batch_index, action].shape)
            print(reward.shape)
            print(pred_next[batch_index, action_max].shape)
            print((1-term).shape)'''
            
            target[batch_index, action] = reward + self.gamma*pred_next[batch_index, action_max]*(1-term)
            
            #current = self.dqn(state).gather(1, action.long())
            
            '''print("old_state",state)
            print("issue", issue)
            print("reward", reward)
            print("action", action)
            print("terminal", term )
            
            print("target0", self.dqn(issue).max(1).values.unsqueeze(1))
            print("current", self.dqn(state).gather(1, action.long()))
            print("target",target)'''
            
            loss = self.loss(current_pred, target)
            loss.backward()
            self.optimizer.step()
            
            # Eventually reduce the exploration rate
            
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_rate, self.exploration_min)
    
    def save(self, name):
        torch.save(self.dqn_validation.state_dict(), name+ "/DQN.pt")  
        torch.save(self.rem_states,  name+ "/rem_states.pt")
        torch.save(self.rem_actions, name+ "/rem_actions.pt")
        torch.save(self.rem_issues, name+ "/rem_issues.pt")
        torch.save(self.rem_rewards, name+ "/rem_rewards.pt")
        torch.save(self.rem_terminals,   name+ "/rem_terminals.pt")
        with open(name+ "/ending_position.pkl", "wb") as f:
            pickle.dump(self.current_position, f)
        with open(name+ "/num_in_queue.pkl", "wb") as f:
            pickle.dump(self.is_full, f)
    
    def update_params(self):
        self.dqn_target.load_state_dict(self.dqn_validation.state_dict())
    

    def get_exploration(self):
        return self.exploration_rate