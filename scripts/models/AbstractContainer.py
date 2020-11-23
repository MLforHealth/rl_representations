from abc import ABC,abstractmethod 
import torch
from .common import pearson_correlation

class AbstractContainer(ABC): 
    '''
    Abstract class for a model container.
    Contains self.gen (defined after calling make_encoder)
             self.pred (defined after calling make_decoder)
    The loop function runs one batch of inputs through the encoder and decoder, and returns the loss.
    Models other than [AE, AIS, RNN] should overload the loop function.
    '''
    @abstractmethod
    def __init__(self, device, **kwargs): 
        pass
    
    @abstractmethod
    def make_encoder(self, **kwargs): 
        pass
    
    @abstractmethod
    def make_decoder(self, **kwargs): 
        pass
    
    def loop(self, obs, dem, actions, scores, l, max_length, context_input, corr_coeff_param, device='cpu', **kwargs):
        '''This loop through the training and validation data is the general template for AIS, RNN, etc'''
        # Split the observations 
        autoencoder = kwargs['autoencoder']
        cur_obs, next_obs = obs[:,:-1,:], obs[:,1:,:]
        cur_dem = dem[:,:-1,:]
        # cur_scores, next_scores = scores[:,:-1,:], scores[:,1:,:] # I won't need the "next scores"
        mask = (cur_obs ==0).all(dim=2) # Compute mask for extra appended rows of observations (all zeros along dim 2)

        # This concatenates an empty action with the first observation and shifts all actions 
        # to the next observation since we're interested in pairing obs with previous action
        if context_input:
            hidden_states = self.gen(torch.cat((cur_obs, cur_dem, torch.cat((torch.zeros((obs.shape[0],1,actions.shape[-1])).to(device),actions[:,:-2,:]),dim=1)),dim=-1))
        else:
            hidden_states = self.gen(torch.cat((cur_obs, torch.cat((torch.zeros((obs.shape[0],1,actions.shape[-1])).to(device), actions[:,:-2,:]),dim=1)), dim=-1))

        if autoencoder == 'RNN':
            pred_obs = self.pred(hidden_states)
        else:
            pred_obs = self.pred(torch.cat((hidden_states,actions[:,:-1,:]),dim=-1))

        # Calculate the correlation between the hidden parameters and the acuity score (For now we'll use SOFA--idx 0)
        corr_loss = pearson_correlation(hidden_states[~mask], scores[:,:-1,:][~mask], device=device)
        temp_loss = -torch.distributions.MultivariateNormal(pred_obs, torch.eye(pred_obs.shape[-1]).to(device)).log_prob(next_obs)
        mse_loss = sum(temp_loss[~mask])
        loss_pred = mse_loss - corr_coeff_param*corr_loss.sum() # We only want to keep the relevant rows of the loss, sum them up! We then add the scaled correlation coefficient

        return loss_pred, mse_loss, hidden_states