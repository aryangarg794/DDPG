
import torch.nn as nn
import numpy as np
import torch
import gymnasium as gym
import torch.nn.functional as F

from copy import deepcopy
from model_utils import fanin_init, OUActionNoise
from utils.replay import ReplayBuffer


class Critic(nn.Module):
    
    def __init__(
        self, 
        env: gym.Env,
        hidden_layers: list = list([400, 300]), 
        *args, 
        **kwargs
    ):
        super(Critic, self).__init__(*args, **kwargs)
        
        self.env = env
        
        self.layers = nn.Sequential(
            nn.Linear(np.prod(self.env.observation_space.shape) + np.prod(self.env.action_space.shape[0]), hidden_layers[0]), # note: this assumes obs spaces of shape (N,) ie 1d
            # nn.LayerNorm(hidden_layers[0]),
            nn.ReLU(),
            
            nn.Linear(hidden_layers[0] , hidden_layers[1]), 
            # nn.LayerNorm(hidden_layers[1]),
            nn.ReLU(),
            
            nn.Linear(hidden_layers[1], 1), 
        )
        
        
        self.layers[0].weight.data = fanin_init(self.layers[0].weight.data.size())
        self.layers[2].weight.data = fanin_init(self.layers[2].weight.data.size())
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(
        self, 
        obs: torch.Tensor,
        action: torch.Tensor
    ):
        rep_and_act = torch.hstack((obs, action))
        q_val = self.layers(rep_and_act)
        return q_val
    
        


class Actor(nn.Module):
    
    def __init__(
        self, 
        env: gym.Env,
        hidden_layers: list = list([400, 300]),
        sigma: float = 0.2,
        *args, 
        **kwargs
    ):
        super(Actor, self).__init__(*args, **kwargs)
        
        self.env = env
        self.high = self.env.action_space.high[0]
        self.low = self.env.action_space.low[0]
        self.shape = self.env.action_space.shape[0]
        self.scale = sigma
        
        self.layers = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], hidden_layers[0]),
            # nn.LayerNorm(hidden_layers[0]),
            nn.ReLU(),
            
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            # nn.LayerNorm(hidden_layers[1]),
            nn.ReLU(),
            
            nn.Linear(hidden_layers[1], self.shape),
            nn.Tanh(),
        )
        
        self.ounoise = OUActionNoise(mean=np.zeros(1), std_deviation=float(sigma) * np.ones(1))
        
        
        self.layers[0].weight.data = fanin_init(self.layers[0].weight.data.size())
        self.layers[2].weight.data = fanin_init(self.layers[2].weight.data.size())
        self.layers[-2].weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(
        self, 
        obs: torch.Tensor,
    ): 
        out = self.layers(obs) # between [-1, 1]
        out = self.high * out
        return out
    
    def sample_actions(
        self, 
        obs: torch.Tensor,
    ): 
        self.eval()
        with torch.no_grad():
            actions = self(obs)
            out = actions.cpu().detach().numpy() + np.random.normal(scale=self.scale * self.high, size=self.shape) 
            out = out.clip(self.low, self.high).reshape(self.shape)
        self.train()
        return out
    
class DDPGAgent:
    
    def __init__(
        self,
        env: gym.Env, 
        tau: float = 0.001,
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001, 
        wd: float = 0.0,
        device: str = 'cpu',
        ou_sigma: float = 0.1, 
        gamma: float = 0.99, 
        capacity: int = 100000, 
    ):
        self.actor = Actor(env, sigma=ou_sigma).to(device)
        self.critic = Critic(env).to(device)
    
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=wd)
        
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        
        self.gamma = gamma
        self.env = env
        self.tau = tau
        self.device = device
        self.buffer = ReplayBuffer(self.env.observation_space.shape[0], 
                                   self.env.action_space.shape[0], capacity, self.device)
        
        
    def update(
        self,
        batch_size: int = 32
    ): 
        with torch.no_grad():
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self.buffer.sample(
                batch_size)

            batch_rewards = batch_rewards.view(-1, 1)
            batch_dones = batch_dones.view(-1, 1) 
            batch_actions = batch_actions.view(-1, np.prod(self.env.action_space.shape)) 
            
            target_actions = self.actor_target(batch_next_obs)
            q_targets = self.critic_target(batch_next_obs, target_actions)
            td_target = batch_rewards + self.gamma * q_targets * (1 - batch_dones.float())
            
           
        q_values = self.critic(batch_obs, batch_actions) 
        loss_critic = F.mse_loss(q_values, td_target)
        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        loss_actor = -self.critic(batch_obs, self.actor(batch_obs)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        return loss_critic
    
    def soft_update(self):
        # soft target update 
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data) 