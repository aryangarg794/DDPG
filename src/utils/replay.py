import random
import torch
import numpy as np

from collections import deque


class ReplayBuffer:
    
    def __init__(self, buffer_len=5000):
        self.store = {
            'states' : deque(maxlen=buffer_len),
            'actions' : deque(maxlen=buffer_len),
            'rewards' : deque(maxlen=buffer_len),
            'next_states' : deque(maxlen=buffer_len),
            'dones' : deque(maxlen=buffer_len)
        }
    
    def update(
        self, 
        state, 
        action, 
        reward, 
        next_state,
        done
    ):
        self.store['states'].append(state)
        self.store['actions'].append(action)
        self.store['rewards'].append(reward)
        self.store['next_states'].append(next_state)
        self.store['dones'].append(done)
    
    def sample(self, buffer_size, device):
        states = random.choices(self.store['states'], k=buffer_size)
        actions = random.choices(self.store['actions'], k=buffer_size)
        rewards = random.choices(self.store['rewards'], k=buffer_size)
        next_states = random.choices(self.store['next_states'], k=buffer_size)
        dones = random.choices(self.store['dones'], k=buffer_size)
        
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(actions), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(dones), dtype=torch.bool, device=device)
        )
        
    def __len__(self):
        return len(self.store['states'])