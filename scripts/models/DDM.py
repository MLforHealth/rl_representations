import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .AbstractContainer import AbstractContainer
from .common import get_dynamics_losses, weights_init, pearson_correlation

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device
    
    def make_encoder(self, *args, **kwargs):
        self.gen = DDM_Encoder(*args, **kwargs).to(self.device)
        return self.gen

    def make_decoder(self, state_dim, hidden_size):
        self.hidden_size = hidden_size
        self.pred = DDM_Decoder(state_dim, hidden_size).to(self.device)
        return self.pred   
    
    def make_dyn(self, *args, **kwargs):
        self.dyn = DDM_D_Module(*args, **kwargs).to(self.device)
        return self.dyn
    
    def loop(self, obs, dem, actions, scores, l, max_length, context_input, corr_coeff_param, device, train_loss=0, dec_loss=0, inv_loss=0, model_loss=0, recon_loss=0, forward_loss=0, **kwargs):
        '''This loop through the training and validation data is for the specific modules present in DDM'''
        # Initialize hidden states for the LSTM layer
        cx_d = torch.zeros(1, obs.shape[0], self.hidden_size).to(device)
        hx_d = torch.zeros(1, obs.shape[0], self.hidden_size).to(device)
        
        cur_obs, next_obs = obs[:,:-1,:], obs[:,1:,:]
        cur_dem, next_dem = dem[:,:-1,:], dem[:,1:,:]
        cur_actions = actions[:,:-1,:]

        mask = (cur_obs == 0).all(dim=2) # Compute mask for extra appended rows of observations (all zeros along dim 2)

        ''' Adapted from train_dynamics() and forward_planning()
        out of: https://github.com/facebookresearch/ddr/blob/master/train_dynamics_module.py
        '''
        if context_input:
            z = self.gen(torch.cat((cur_obs, cur_dem), dim=-1))
            z_prime = self.gen(torch.cat((next_obs, next_dem), dim=-1))
        else:
            z = self.gen(cur_obs)
            z_prime = self.gen(next_obs)
        
        s_hat = self.pred(z)
        z_prime_hat, a_hat, _ = self.dyn((z,z_prime,cur_actions,(hx_d,cx_d)))
        s_prime_hat = self.pred(z_prime_hat)

        # Loss in predicting distribution of next observation
        r_loss, m_loss, d_loss, i_loss, f_loss = get_dynamics_losses(
            cur_obs[~mask], s_hat[~mask], next_obs[~mask], s_prime_hat[~mask], z_prime[~mask], z_prime_hat[~mask],
            a_hat[~mask], cur_actions[~mask], discrete=False)

        inv_loss += i_loss
        dec_loss += d_loss
        forward_loss += f_loss
        recon_loss += r_loss
        model_loss += m_loss

        corr_loss = pearson_correlation(z[~mask], scores[:,:-1,:][~mask], device=device)

        temp_loss = -torch.distributions.MultivariateNormal(s_prime_hat, torch.eye(s_prime_hat.shape[-1]).to(device)).log_prob(next_obs)
        loss_pred = sum(temp_loss[~mask]) # We only want to keep the relevant rows of the loss, sum them up!
        
        return (train_loss, dec_loss, inv_loss, model_loss, recon_loss, forward_loss, corr_loss, loss_pred), None, None # compatability with other functions

    
'''
The following three model classes are pulled from https://github.com/facebookresearch/ddr/blob/master/model.py, 
with minor changes, a repo built by Amy Zhang to accompany her paper: https://arxiv.org/abs/1804.10689
'''
class DDM_Encoder(torch.nn.Module):
    def __init__(self, obs_space, dim, context_input=False, context_dim=0):
        """
        architecture should be input, so that we can pass multiple jobs !
        """
        super(DDM_Encoder, self).__init__()
        if context_input:
            self.linear1 = nn.Linear(obs_space+context_dim, dim)
        else:
            self.linear1 = nn.Linear(obs_space, dim)
        self.linear2 = nn.Linear(dim, 32 * 3 * 3)
        self.fc = nn.Linear(32 * 3 * 3, dim)
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        # why elu and not relu ?
        x = F.elu(self.linear1(inputs))
        x = F.elu(self.linear2(x))

        x = F.tanh(self.fc(x))

        return x

class DDM_Decoder(torch.nn.Module):
    def __init__(self, obs_space, dim):
        super(DDM_Decoder, self).__init__()
        self.fc = nn.Linear(dim, 32 * 3 * 3)
        self.linear1 = nn.Linear(32 * 3 * 3, dim)
        self.linear2 = nn.Linear(dim, obs_space)
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = F.elu(self.fc(inputs))
        
        x = F.elu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class DDM_D_Module(torch.nn.Module):
    def __init__(self, action_space, dim, discrete=False):
        super(DDM_D_Module, self).__init__()
        self.dim = dim
        self.discrete = discrete

        self.za_embed = nn.Linear(2 * dim, dim)
        self.lstm_dynamics = nn.LSTM(dim, dim)
        self.z_embed = nn.Linear(dim, dim)

        self.inv = nn.Linear(2 * dim, dim)
        self.inv2 = nn.Linear(dim, action_space)

        self.action_linear = nn.Linear(action_space, dim)
        self.action_linear2 = nn.Linear(dim, dim)
        self.apply(weights_init)

        self.lstm_dynamics.bias_ih_l0.data.fill_(0)
        self.lstm_dynamics.bias_hh_l0.data.fill_(0)

        self.train()

    def forward(self, inputs):
        z, z_prime, actions, (hx_d, cx_d) = inputs
#         z = z.view(-1, self.dim)

        a_embedding = F.elu(self.action_linear(actions))
        a_embedding = self.action_linear2(a_embedding)
        
        za_embedding = self.za_embed(
            torch.cat([z, a_embedding], dim=-1))

        za_embedding = za_embedding.permute(1,0,2)
        output, _ = self.lstm_dynamics(za_embedding, (hx_d, cx_d))
        output = output.permute(1, 0, 2)
        z_prime_hat = F.tanh(self.z_embed(output))

        # decode the action
#         if z_prime is not None:
#             z_prime = z_prime.view(-1, self.dim)
        if z_prime is None:
            z_prime = z_prime_hat
        a_hat = F.elu(self.inv(torch.cat([z, z_prime], dim=-1)))
        a_hat = self.inv2(a_hat)
        return z_prime_hat, a_hat, (hx_d, cx_d)    