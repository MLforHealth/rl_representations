import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    if n_layers == 0:
        return nn.Linear(n_inputs, n_outputs)
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers-1):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

def pearson_correlation(latents, scores, device='cpu'):
    '''
    Calculate the pearson correlation between the latent vectors and the provided scores
    '''
    vx = latents - latents.mean(0)[None,:]
    vy = (scores - scores.mean(0)[None,:]) + 1e-6 

    corr_outputs = (vx[...,None] * vy.unsqueeze(1)).sum(0) / (torch.sqrt((vx ** 2).sum(0))[:,None] * torch.sqrt((vy ** 2).sum(0))[None,:])

    return corr_outputs

def mask_from_lengths(lens, max_len, device):
    return torch.arange(max_len).expand(len(lens), max_len).to(device) < lens.unsqueeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('GRUCell') != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
                
def get_dynamics_losses(s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat,
                        a_hat, curr_actions, discrete=False):
    
    # reconstruction loss
    recon_loss = F.mse_loss(s_hat, s)

    # next state prediction loss
    model_loss = F.mse_loss(s_prime_hat, s_prime)

    # net decoder loss
    dec_loss = (F.mse_loss(s_hat, s) + F.mse_loss(s_prime_hat, s_prime))

    # action reconstruction loss
    if discrete:
        a_hat = F.log_softmax(a_hat)

    inv_loss = F.mse_loss(a_hat, curr_actions)

    # representation space constraint
    forward_loss = F.mse_loss(z_prime_hat, z_prime.detach())
    return recon_loss, model_loss, dec_loss, inv_loss, forward_loss