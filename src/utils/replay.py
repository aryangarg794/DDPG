import torch
import numpy as np


# code based on https://github.com/XinJingHao/DDPG-Pytorch?tab=readme-ov-file
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_len, device='cpu'):
        self.capacity = buffer_len
        self.device = device
        self.pointer = 0
        self.size = 0

        self.states = torch.zeros((self.capacity, state_dim) ,dtype=torch.float,device=self.device)
        self.actions = torch.zeros((self.capacity, action_dim) ,dtype=torch.float,device=self.device)
        self.rewards = torch.zeros((self.capacity, 1) ,dtype=torch.float,device=self.device)
        self.next_states = torch.zeros((self.capacity, state_dim) ,dtype=torch.float,device=self.device)
        self.dones = torch.zeros((self.capacity, 1) ,dtype=torch.bool,device=self.device)

    def update(
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done
    ):

        self.states[self.pointer] = torch.as_tensor(state).to(self.device)
        self.actions[self.pointer] = torch.as_tensor(action).to(self.device) 
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = torch.as_tensor(next_state).to(self.device)
        self.dones[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.capacity 
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return (
            self.states[ind], 
            self.actions[ind], 
            self.rewards[ind], 
            self.next_states[ind], 
            self.dones[ind]
        )
        
    def __len__(self):
        return len(self.states)