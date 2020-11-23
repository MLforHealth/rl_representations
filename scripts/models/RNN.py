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
        self.gen = baseRNN_generate(hidden_size, state_dim, num_actions, context_input, context_dim).to(self.device)
        return self.gen

    def make_decoder(self, hidden_size, state_dim, num_actions):
        self.pred = baseRNN_predict(hidden_size, state_dim, num_actions).to(self.device)
        return self.pred   

class baseRNN_generate(nn.Module):
    def __init__(self, h_size, obs_dim, num_actions, context_input=False, context_dim=0):
        super(baseRNN_generate,self).__init__()
        if context_input:
            self.l1 = nn.Linear(obs_dim + context_dim + num_actions, 64)
        else:
            self.l1 = nn.Linear(obs_dim + num_actions,64)
        self.l2 = nn.Linear(64,128)
        self.l3 = nn.GRU(128,h_size)
        self.apply(weights_init)
    def forward(self,x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = x.permute(1,0,2)
        h, _ = self.l3(x)
        return h.permute(1,0,2)

class baseRNN_predict(nn.Module):
    def __init__(self, h_size, obs_dim, num_actions, context_input=False):
        super(baseRNN_predict,self).__init__()
        self.l1 = nn.Linear(h_size, 64)
        self.l2 = nn.Linear(64,128)
        self.l3 = nn.Linear(128,obs_dim)
        self.apply(weights_init)
    def forward(self,h):
        h = torch.relu(self.l1(h))
        h = torch.relu(self.l2(h))
        obs = self.l3(h)
        return obs
    
    