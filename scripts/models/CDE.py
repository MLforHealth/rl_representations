import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .NeuralCDE.metamodel import NeuralCDE
from .NeuralCDE.vector_fields import SingleHiddenLayer, FinalTanh
from .common import create_net, pearson_correlation, mask_from_lengths
from .AbstractContainer import AbstractContainer

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device
    
    def make_encoder(self, input_channels, hidden_channels, hidden_hidden_channels = 50, num_hidden_layers = 4):
        vector_field = FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                                hidden_hidden_channels=hidden_hidden_channels,
                                                num_hidden_layers=num_hidden_layers)
        self.gen = NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels, initial=True).to(self.device)
        return self.gen

    def make_decoder(self, latent_dim, output_channels, n_layers = 3, n_units = 100):
        self.pred = create_net(n_inputs = latent_dim, n_outputs = output_channels, n_layers = n_layers, n_units = n_units).to(self.device)
        return self.pred
    
    def loop(self, ob, dem, ac, scores, l, max_length, context_input, corr_coeff_param = 0.0, device = 'cuda', **kwargs):
        coefs = kwargs['coefs']
        idx = kwargs['idx']
        
        targets = torch.cat((ob[:, 1:, :], torch.zeros(ob.shape[0], 1, ob.shape[-1]).to(device)), dim = 1)
        pred_mask = mask_from_lengths(l, max_length+1, device = device)[:, 1:]
        times = torch.arange(coefs[0].shape[1]+1, device=device).float()    
        
        coeffs_batch = [i[idx].to(device) for i in coefs]
                        
        hidden = self.gen(times, coeffs_batch, final_index = -1, stream = True)[:, :max_length, :]
        output = self.pred(hidden)
                
        total_loss = F.mse_loss(targets[pred_mask], output[pred_mask])
        mse_loss = total_loss.item()
        
        if corr_coeff_param > 0:
            corr_loss = pearson_correlation(hidden[pred_mask], scores[pred_mask], device=device).mean()
            total_loss += (-corr_coeff_param * corr_loss)
            
        return total_loss, mse_loss, hidden