import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .AbstractContainer import AbstractContainer
from .common import weights_init

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device
    
    def make_encoder(self, hidden_size, state_dim, num_actions, context_input, context_dim):
        self.gen = baseAE_generate(hidden_size, state_dim, num_actions, context_input, context_dim).to(self.device)
        return self.gen

    def make_decoder(self, hidden_size, state_dim, num_actions):
        self.pred = baseAE_predict(hidden_size, state_dim, num_actions).to(self.device)
        return self.pred   
    
class baseAE_generate(nn.Module):
    def __init__(self,h_size, obs_dim, num_actions, context_input=False, context_dim=0):
        super(baseAE_generate,self).__init__()
        if context_input:
            self.l1 = nn.Linear(obs_dim+context_dim+num_actions, 64)
        else:
            self.l1 = nn.Linear(obs_dim + num_actions, 64)
        self.l2 = nn.Linear(64,128)
        self.l3 = nn.Linear(128, h_size)
        self.apply(weights_init)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        h = self.l3(x)
        return h

class baseAE_predict(nn.Module):
    def __init__(self, h_size, obs_dim, num_actions, context_input=False, context_dim=0):
        super(baseAE_predict, self).__init__()
        self.l1 = nn.Linear(h_size+num_actions, 64)
        self.l2 = nn.Linear(64,128)
        self.l3 = nn.Linear(128,obs_dim)
        self.apply(weights_init)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        obs = self.l3(x)
        return obs