import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'latent_ode/') 
from lib.latent_ode import LatentODE
from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import Encoder_z0_ODE_RNN, Decoder
from .AbstractContainer import AbstractContainer
from .common import create_net, mask_from_lengths, pearson_correlation
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device          
    
    def make_encoder(self, input_dim, latent_dim, odernn_hypers):
        self.gen = get_odernn_encoder(input_dim, latent_dim, odernn_hypers, self.device).to(self.device)
        return self.gen

    def make_decoder(self, hidden_size, state_dim, decoder_n_layers, decoder_n_units):
        self.pred = create_net(hidden_size, state_dim, 
                                       decoder_n_layers, 
                                   decoder_n_units).to(self.device)
        return self.pred
    
    def loop(self, ob, dem, ac, scores, l, max_length, context_input, corr_coeff_param = 0.0, device = 'cuda', **kwargs):
        ac_shifted = torch.cat((torch.zeros(ac.shape[0], 1, ac.shape[-1]).to(device), ac[:, :-1, :]), dim = 1) # action at t-1
        observed_mask = mask_from_lengths(l, max_length, device).unsqueeze(-1)
        ac_shifted[observed_mask.expand(-1, -1, ac.shape[-1]) == 0] = 0 # if no obs at time t, zero out actions at t-1

        if context_input:
            obs_data = torch.cat((ob, dem, ac_shifted), dim = -1)
        else:
            obs_data = torch.cat((ob, ac_shifted), dim = -1)
        
        obs_mask = observed_mask.expand(*obs_data.shape)
    
        targets = torch.cat((ob[:, 1:, :], torch.zeros(ob.shape[0], 1, ob.shape[-1]).to(device)), dim = 1)
        pred_mask = mask_from_lengths(l, max_length+1, device = device)[:, 1:]
        
        times = torch.arange(max_length, device=device).float()    
                    
        _, _, hidden, _ = self.gen.run_odernn(torch.cat((obs_data, obs_mask), dim = -1) , times, 
                    run_backwards = False, save_info = False)
        hidden = hidden.permute(0,2,1,3).squeeze(0)
        output = self.pred(hidden)
        
        total_loss = F.mse_loss(targets[pred_mask], output[pred_mask])
        mse_loss = total_loss.item()
        
        if corr_coeff_param > 0:
            corr_loss = pearson_correlation(hidden[pred_mask], scores[pred_mask], device=device).mean()
            total_loss += (-corr_coeff_param * corr_loss)

        return total_loss, mse_loss, hidden

def get_odernn_encoder(input_dim, latent_dim, odernn_hypers, device):
    ode_func_net = create_net(latent_dim, latent_dim,
            n_layers = odernn_hypers['odefunc_n_layers'], n_units = odernn_hypers['odefunc_n_units'])

    ode_func = ODEFunc(
                input_dim = input_dim * 2, 
                latent_dim = latent_dim,
                ode_func_net = ode_func_net,
                device = device).to(device)

    diffeq_solver = DiffeqSolver(input_dim * 2, ode_func, 'dopri5', latent_dim, 
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

    encoder = Encoder_z0_ODE_RNN(latent_dim, input_dim * 2, diffeq_solver, 
            z0_dim = latent_dim, n_gru_units = odernn_hypers['encoder_gru_units'], device = device).to(device)
    
    return encoder
    