import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .common import create_net, pearson_correlation, mask_from_lengths
from .AbstractContainer import AbstractContainer
import signatory

class ModelContainer(AbstractContainer):
    def __init__(self, device):
        self.device = device
    
    def make_encoder(self, input_dim, latent_dim, gru_n_layers, augment_chs):
        self.gen = DST_Encoder(input_dim, latent_dim, gru_n_layers, augment_chs).to(self.device)
        return self.gen

    def make_decoder(self, latent_dim, output_dim, decoder_hidden_units):
        self.pred = DST_Decoder(latent_dim, output_dim, decoder_hidden_units).to(self.device)
        return self.pred
    
    def loop(self, ob, dem, ac, scores, l, max_length, context_input, corr_coeff_param = 0.0, device = 'cpu', **kwargs):
        ac = torch.cat((torch.zeros(ac.shape[0], 1, ac.shape[-1]).to(device), ac[:, :-1, :]), dim = 1) # action at t-1        
        if context_input:
            obs_data = torch.cat((ob, dem, ac), dim = -1)
        else:           
            obs_data = torch.cat((ob, ac), dim = -1)

        hidden = self.gen(obs_data)
        output = self.pred(hidden)

        targets = torch.cat((ob[:, 1:, :], torch.zeros(ob.shape[0], 1, ob.shape[-1]).to(device)), dim = 1)
        mask = mask_from_lengths(l, max_length+1, device = device)[:, 1:]

        total_loss = F.mse_loss(targets[mask], output[mask])
        mse_loss = total_loss.item()
        if corr_coeff_param > 0:
            corr_loss = pearson_correlation(hidden[mask], scores[mask], device=device).mean()
            total_loss += (-corr_coeff_param * corr_loss)

        return total_loss, mse_loss, hidden
    
class DST_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, gru_n_layers = 2, augment_chs = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.augment1 = signatory.Augment(in_channels=input_dim,
                          layer_sizes=(32, augment_chs),
                          kernel_size=1,
                          include_original=True,
                          include_time=True)

        self.signature1 = signatory.Signature(depth=2,
                             stream=True)

        sig_channels1 = signatory.signature_channels(channels=input_dim + augment_chs + 1,
                                              depth=2)

        self.gru = nn.GRU(input_size = sig_channels1, hidden_size = latent_dim, num_layers = gru_n_layers, batch_first = True)

    def forward(self, x):
        return self.gru(self.signature1(self.augment1(x), basepoint = True))[0]   
    
    
class DST_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, decoder_hidden_units = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_hidden_units = decoder_hidden_units
        self.augment2 = signatory.Augment(in_channels=latent_dim,
                                  layer_sizes=(64, 32),
                                  kernel_size=1,
                                  include_original=False,
                                  include_time=False)

        self.signature2 = signatory.Signature(depth=2,
                             stream=True)

        sig_channels2 = signatory.signature_channels(channels= 32,
                                              depth=2)
        
        self.linear1 = nn.Linear(in_features = sig_channels2, out_features = decoder_hidden_units)
        self.linear2 = nn.Linear(in_features = decoder_hidden_units, out_features = output_dim)
                

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(self.signature2(self.augment2(x), basepoint = True))))
    