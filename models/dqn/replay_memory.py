from collections import deque
import random
import torch
import numpy as np

class Memory():
    def __init__(self, max_mem_size) -> None:
        
        self.memory = deque([], maxlen= max_mem_size)
        
        self._size = 0
        self.max_size = max_mem_size
        
    def append(self, state, action, reward, issue, terminal):
        self._size = min(self.size +1, self.max_size)
        
        self.memory.append([state, action, reward, issue, terminal])
    
    def compute_batch(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        issue_batch = []
        terminal_batch = []
        
        for i in range(batch_size):
            state_batch.append(batch[i][0])
            action_batch.append(batch[i][1])
            reward_batch.append(batch[i][2])
            issue_batch.append(batch[i][3])
            terminal_batch.append(batch[i][4])
        
        state_batch = torch.tensor(np.array(state_batch)).float()
        action_batch = torch.tensor(action_batch).float()
        reward_batch = torch.tensor(reward_batch).float()
        issue_batch = torch.tensor(np.array(issue_batch)).float()
        terminal_batch = torch.tensor(terminal_batch).float()

        
        return state_batch, action_batch, reward_batch, issue_batch, terminal_batch
    
    @property
    def size(self):
        return self._size