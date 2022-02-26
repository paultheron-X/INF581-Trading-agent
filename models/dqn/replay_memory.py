from collections import deque
import random
import torch
import numpy as np

class Memory():
    def __init__(self, max_mem_size) -> None:
        
        self.state = deque(maxlen = max_mem_size)
        self.action = deque(maxlen = max_mem_size) 
        self.reward = deque(maxlen = max_mem_size) 
        self.issue = deque(maxlen = max_mem_size) 
        self.terminal = deque(maxlen = max_mem_size)
        
        self._size = 0
        self.max_size = max_mem_size
        
    def append(self, state, action, reward, issue, terminal):
        
        self._size = min(self.size +1, self.max_size)
        
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.issue.append(issue)
        self.terminal.append(terminal)
    
    def compute_batch(self, batch_size):
        
        state_batch = torch.tensor(np.array(random.sample(self.state, batch_size))).float()
        action_batch = torch.tensor(random.sample(self.action, batch_size)).float()
        reward_batch = torch.tensor(random.sample(self.reward, batch_size)).float()
        issue_batch = torch.tensor(np.array(random.sample(self.issue, batch_size))).float()
        terminal_batch = torch.tensor(random.sample(self.terminal, batch_size)).float()
        
        return state_batch, action_batch, reward_batch, issue_batch, terminal_batch
    
    @property
    def size(self):
        return self._size