import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .AbstractContainer import AbstractContainer

class ModelContainer(AbstractContainer):
    def __init__(self, device, ais_gen_model, ais_pred_model):
        self.device = device
        if ais_gen_model == 1:
            self.gen_model = AISGenerate_1
        elif ais_gen_model == 2:
            self.gen_model = AISGenerate_2
        if ais_pred_model == 1:
            self.pred_model = AISPredict_1
        elif ais_pred_model == 2:
            self.pred_model = AISPredict_2            
    
    def make_encoder(self, hidden_size, state_dim, num_actions, context_input, context_dim):
        self.gen = self.gen_model(hidden_size, state_dim, num_actions, context_input, context_dim).to(self.device)
        return self.gen

    def make_decoder(self, hidden_size, state_dim, num_actions):
        self.pred = self.pred_model(hidden_size, state_dim, num_actions).to(self.device)
        return self.pred    

class AISGenerate_1(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions, context_input=False, context_dim=0):
        super(AISGenerate_1, self).__init__()
        if context_input:
            self.l1 = nn.Linear(obs_dim + context_dim + num_actions, 64)
        else:
            self.l1 = nn.Linear(obs_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.GRU(128, ais_size)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = x.permute(1, 0, 2) # Put time sequence first for GRU layer
        h, _ = self.l3(x)
        return h.permute(1,0,2) # Put the batch size first again...

class AISGenerate_2(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions, context_input=False, context_dim=0):
        super(AISGenerate_2, self).__init__()
        if context_input:
            self.l1 = nn.Linear(obs_dim + context_dim + num_actions, 64)
        else:
            self.l1 = nn.Linear(obs_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.GRUCell(64, ais_size)
    def forward(self, x, h):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        h = self.l4(x, h)
        return h

class AISPredict_1(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions, context_input=False, context_dim=0):
        super(AISPredict_1, self).__init__()
        self.l1 = nn.Linear(ais_size + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, obs_dim)
        
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        obs = self.l3(x)
        return obs

class AISPredict_2(nn.Module):
    def __init__(self, ais_size, obs_dim, num_actions, context_input=False, context_dim=0):
        super(AISPredict_2, self).__init__()
        self.l1 = nn.Linear(ais_size + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, obs_dim)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        obs = self.l4(x)
        return obs