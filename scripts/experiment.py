'''
This module defines the Experiment class that intializes, trains, and evaluates a Recurrent autoencoder.

The central focus of this class is to develop representations of sequential patient states in acute clinical settings.
These representations are learned through an auxiliary task of predicting the subsequent physiological observation but 
are also used to train a treatment policy via offline RL. The specific policy learning algorithm implemented through this
module is the discretized form of Batch Constrained Q-learning [Fujimoto, et al (2019)]

This module was designed and tested for use with a Septic patient cohort extracted from the MIMIC-III (v1.4) database. It is
assumed that the data used to create the Dataloaders in lines 174, 180 and 186 is patient and time aligned separate sequences 
of:
    (1) patient demographics
    (2) observations of patient vitals, labs and other relevant tests
    (3) assigned treatments or interventions
    (4) how long each patient trajectory is
    (5) corresponding patient acuity scores, and
    (6) patient outcomes (here, binary - death vs. survival)

The cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:
 - The code for the AIS approach and general framework we build from was developed by Jayakumar Subramanian

'''
import numpy as np
import pandas as pd
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from itertools import chain

import signatory

from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from utils import one_hot, load_cde_data, process_cde_data, ReplayBuffer
import os
import copy
import pickle

from dBCQ_utils import *

from models import AE, AIS, CDE, DST, DDM, RNN, ODERNN
from models.common import get_dynamics_losses, pearson_correlation, mask_from_lengths

class Experiment(object): 
    def __init__(self, domain, train_data_file, validation_data_file, test_data_file, minibatch_size, rng, device,
                 behav_policy_file_wDemo, behav_policy_file,
                context_input=False, context_dim=0, drop_smaller_than_minibatch=True, 
                folder_name='/Name', autoencoder_saving_period=20, resume=False, sided_Q='negative',  
                autoencoder_num_epochs=50, autoencoder_lr=0.001, autoencoder='AIS', hidden_size=16, ais_gen_model=1, 
                ais_pred_model=1, embedding_dim=4, state_dim=42, num_actions=25, corr_coeff_param=10, dst_hypers = {},
                 cde_hypers = {}, odernn_hypers = {},  **kwargs):
        '''
        We assume discrete actions and scalar rewards!
        '''

        self.rng = rng
        self.device = device
        self.train_data_file = train_data_file
        self.validation_data_file = validation_data_file
        self.test_data_file = test_data_file
        self.minibatch_size = minibatch_size
        self.drop_smaller_than_minibatch = drop_smaller_than_minibatch
        self.autoencoder_num_epochs = autoencoder_num_epochs 
        self.autoencoder = autoencoder
        self.autoencoder_lr = autoencoder_lr
        self.saving_period = autoencoder_saving_period
        self.resume = resume
        self.sided_Q = sided_Q
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.corr_coeff_param = corr_coeff_param

        self.context_input = context_input # Check to see if we'll one-hot encode the categorical contextual input
        self.context_dim = context_dim # Check to see if we'll remove the context from the input and only use it for decoding
        self.hidden_size = hidden_size
        
        if self.context_input:
            self.input_dim = self.state_dim + self.context_dim + self.num_actions
        else:
            self.input_dim = self.state_dim + self.num_actions
        
        self.autoencoder_lower = self.autoencoder.lower()
        self.data_folder = folder_name + f'/{self.autoencoder_lower}_data'
        self.checkpoint_file = folder_name + f'/{self.autoencoder_lower}_checkpoints/checkpoint.pt'
        if not os.path.exists(folder_name + f'/{self.autoencoder_lower}_checkpoints'):
            os.mkdir(folder_name + f'/{self.autoencoder_lower}_checkpoints')
        if not os.path.exists(folder_name + f'/{self.autoencoder_lower}_data'):
            os.mkdir(folder_name + f'/{self.autoencoder_lower}_data')
        self.store_path = folder_name
        self.gen_file = folder_name + f'/{self.autoencoder_lower}_data/{self.autoencoder_lower}_gen.pt'
        self.pred_file = folder_name + f'/{self.autoencoder_lower}_data/{self.autoencoder_lower}_pred.pt'
        
        if self.autoencoder == 'AIS':          
            self.container = AIS.ModelContainer(device,  ais_gen_model, ais_pred_model)
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.num_actions, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.num_actions)
            
        elif self.autoencoder == 'AE':
            self.container = AE.ModelContainer(device)
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.num_actions, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.num_actions)
              
        elif self.autoencoder == 'DST':
            self.dst_hypers = dst_hypers  
            self.container = DST.ModelContainer(device)            
            self.gen = self.container.make_encoder(self.input_dim, self.hidden_size, gru_n_layers = self.dst_hypers['gru_n_layers'],
                                     augment_chs = self.dst_hypers['augment_chs'])
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.dst_hypers['decoder_hidden_units'])    

        elif self.autoencoder == 'DDM':
            self.container = DDM.ModelContainer(device)   
            
            self.gen = self.container.make_encoder(self.state_dim, self.hidden_size, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.state_dim,self.hidden_size)
            self.dyn = self.container.make_dyn(self.num_actions,self.hidden_size)
            self.all_params = chain(self.gen.parameters(), self.pred.parameters(), self.dyn.parameters())
            
            self.inv_loss_coef = 10
            self.dec_loss_coef = 0.1
            self.max_grad_norm = 50

            self.dyn_file = folder_name + '/ddm_data/ddm_dyn.pt'
        
        elif self.autoencoder == 'RNN':
            self.container = RNN.ModelContainer(device)
            
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.num_actions, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.num_actions)
            
        elif self.autoencoder == 'CDE':
            self.cde_hypers = cde_hypers
            
            self.container = CDE.ModelContainer(device)
            self.gen = self.container.make_encoder(self.input_dim + 1, self.hidden_size, hidden_hidden_channels = self.cde_hypers['encoder_hidden_hidden_channels'], num_hidden_layers = self.cde_hypers['encoder_num_hidden_layers'])
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.cde_hypers['decoder_num_layers'] , self.cde_hypers['decoder_num_units'])
            
            
        elif self.autoencoder == 'ODERNN':
            self.odernn_hypers = odernn_hypers    
            self.container = ODERNN.ModelContainer(device)
            
            self.gen = self.container.make_encoder(self.input_dim, self.hidden_size, self.odernn_hypers)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, 
                                       self.odernn_hypers['decoder_n_layers'], 
                                   self.odernn_hypers['decoder_n_units'])
        else:
            raise NotImplementedError

        self.buffer_save_file = self.data_folder + '/ReplayBuffer'
        self.next_obs_pred_errors_file = self.data_folder + '/test_next_obs_pred_errors.pt'
        self.test_representations_file = self.data_folder + '/test_representations.pt'
        self.test_correlations_file = self.data_folder + '/test_correlations.pt'
        self.policy_eval_save_file = self.data_folder + '/dBCQ_policy_eval'
        self.policy_save_file = self.data_folder + '/dBCQ_policy'
        self.behav_policy_file_wDemo = behav_policy_file_wDemo
        self.behav_policy_file = behav_policy_file
        
        
        # Read in the data csv files
        assert (domain=='sepsis')        
        self.train_demog, self.train_states, self.train_interventions, self.train_lengths, self.train_times, self.acuities, self.rewards = torch.load(self.train_data_file)
        train_idx = torch.arange(self.train_demog.shape[0])
        self.train_dataset = TensorDataset(self.train_demog, self.train_states, self.train_interventions,self.train_lengths,self.train_times, self.acuities, self.rewards, train_idx)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.minibatch_size, shuffle=True)

        self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards = torch.load(self.validation_data_file)
        val_idx = torch.arange(self.val_demog.shape[0])
        self.val_dataset = TensorDataset(self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards, val_idx)

        self.val_loader = DataLoader(self.val_dataset, batch_size=self.minibatch_size, shuffle=False)

        self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_acuities, self.test_rewards = torch.load(self.test_data_file)
        test_idx = torch.arange(self.test_demog.shape[0])
        self.test_dataset = TensorDataset(self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_acuities, self.test_rewards, test_idx)

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.minibatch_size, shuffle=False)
        
        # encode CDE data first to save time
        if self.autoencoder == 'CDE':     
            self.train_coefs = load_cde_data('train', self.train_dataset, self.cde_hypers['coefs_folder'],
                                            self.context_input, device)
            self.val_coefs = load_cde_data('val', self.val_dataset, self.cde_hypers['coefs_folder'],
                                            self.context_input, device)
            self.test_coefs = load_cde_data('test', self.test_dataset, self.cde_hypers['coefs_folder'],
                                            self.context_input, device)            
            
    
    def load_model_from_checkpoint(self, checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        self.gen.load_state_dict(checkpoint['{}_gen_state_dict'.format(self.autoencoder.lower())])
        self.pred.load_state_dict(checkpoint['{}_pred_state_dict'.format(self.autoencoder.lower())])
        if self.autoencoder == 'DDM':
            self.dyn.load_state_dict(checkpoint['{}_dyn_state_dict'.format(self.autoencoder.lower())])
        print("Experiment: generator and predictor models loaded.")

    def train_autoencoder(self):
        print('Experiment: training autoencoder')
        device = self.device
        
        if self.autoencoder != 'DDM':
            self.optimizer = torch.optim.Adam(list(self.gen.parameters()) + list(self.pred.parameters()), lr=self.autoencoder_lr, amsgrad=True)
        else:
            self.optimizer = torch.optim.Adam(list(self.gen.parameters()) + list(self.pred.parameters()) + list(self.dyn.parameters()), lr=self.autoencoder_lr, amsgrad=True)

        self.autoencoding_losses = []
        self.autoencoding_losses_validation = []
        
        if self.resume: # Need to rebuild this to resume training for 400 additional epochs if feasible... 
            try:
                checkpoint = torch.load(self.checkpoint_file)
                self.gen.load_state_dict(checkpoint['gen_state_dict'])
                self.pred.load_state_dict(checkpoint['pred_state_dict'])
                if self.autoencoder == 'DDM':
                    self.dyn.load_state_dict(checkpoint['dyn_state_dict'])
                
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                epoch_0 = checkpoint['epoch'] + 1
                self.autoencoding_losses = checkpoint['loss']
                self.autoencoding_losses_validation = checkpoint['validation_loss']
                print('Starting from epoch: {0} and continuing up to epoch {1}'.format(epoch_0, self.autoencoder_num_epochs))
            except:
                epoch_0 = 0
                print('Error loading file, training from default setting. epoch_0 = 0')
        else:
            epoch_0 = 0

        for epoch in range(epoch_0, self.autoencoder_num_epochs):
            epoch_loss = []
            print("Experiment: autoencoder {0}: training Epoch = ".format(self.autoencoder), epoch+1, 'out of', self.autoencoder_num_epochs, 'epochs')

            # Loop through all the train data using the data loader
            for ii, (dem, ob, ac, l, t, scores, rewards, idx) in enumerate(self.train_loader):
                # print("Batch {}".format(ii),end='')
                dem = dem.to(device)  # 5 dimensional vector (Gender, Ventilation status, Re-admission status, Age, Weight)
                ob = ob.to(device)    # 33 dimensional vector (time varying measures)
                ac = ac.to(device) # actions
                l = l.to(device)
                t = t.to(device)
                scores = scores.to(device)
                idx = idx.to(device)
                loss_pred = 0

                # Cut tensors down to the batch's largest sequence length... Trying to speed things up a bit...
                max_length = int(l.max().item())

                # The following losses are for DDM and will not be modified by any other approach
                train_loss, dec_loss, inv_loss = 0, 0, 0
                model_loss, recon_loss, forward_loss = 0, 0, 0                    
                    
                # Set training mode (nn.Module.train()). It does not actually trains the model, but just sets the model to training mode.
                self.gen.train()
                self.pred.train()

                ob = ob[:,:max_length,:]
                dem = dem[:,:max_length,:]
                ac = ac[:,:max_length,:]
                scores = scores[:,:max_length,:]
                
                if self.autoencoder == 'CDE':
                    loss_pred, mse_loss, _ = self.container.loop(ob, dem, ac, scores, l, max_length, self.context_input, corr_coeff_param = self.corr_coeff_param, device = device, coefs = self.train_coefs, idx = idx)
                else:
                    loss_pred, mse_loss, _ = self.container.loop(ob, dem, ac, scores, l, max_length, self.context_input, corr_coeff_param = self.corr_coeff_param, device=device, autoencoder = self.autoencoder)   

                self.optimizer.zero_grad()
                
                if self.autoencoder != 'DDM':
                    loss_pred.backward()
                    self.optimizer.step()
                    epoch_loss.append(loss_pred.detach().cpu().numpy())                
                else:
                    train_loss, dec_loss, inv_loss, model_loss, recon_loss, forward_loss, corr_loss, loss_pred = loss_pred
                    train_loss = forward_loss + self.inv_loss_coef*inv_loss + self.dec_loss_coef*dec_loss - self.corr_coeff_param*corr_loss.sum()
                    train_loss.backward()
                    # Clipping gradients
                    torch.nn.utils.clip_grad_norm(self.all_params, self.max_grad_norm)
                    self.optimizer.step()
                    epoch_loss.append(loss_pred.detach().cpu().numpy())
                                        
            self.autoencoding_losses.append(epoch_loss)
            if (epoch+1)%self.saving_period == 0: # Run validation and also save checkpoint
                
                #Computing validation loss
                epoch_validation_loss = []
                with torch.no_grad():
                    for jj, (dem, ob, ac, l, t, scores, rewards, idx) in enumerate(self.val_loader):

                        dem = dem.to(device)
                        ob = ob.to(device)
                        ac = ac.to(device)
                        l = l.to(device)
                        t = t.to(device)
                        idx = idx.to(device)
                        scores = scores.to(device)
                        loss_val = 0

                        # Cut tensors down to the batch's largest sequence length... Trying to speed things up a bit...
                        max_length = int(l.max().item())                        
                        
                        ob = ob[:,:max_length,:]
                        dem = dem[:,:max_length,:]
                        ac = ac[:,:max_length,:] 
                        scores = scores[:,:max_length,:] 
                        
                        self.gen.eval()
                        self.pred.eval()    
                        
                        if self.autoencoder == 'CDE':
                            loss_val, mse_loss, _ = self.container.loop(ob, dem, ac, scores, l, max_length, corr_coeff_param = 0, device = device, coefs = self.val_coefs, idx = idx)
                        else:
                            loss_val, mse_loss, _ = self.container.loop(ob, dem, ac, scores, l, max_length, self.context_input, corr_coeff_param = 0, device=device, autoencoder = self.autoencoder)                                                 
                        
                        if self.autoencoder in ['DST', 'ODERNN', 'CDE']:
                            epoch_validation_loss.append(mse_loss)
                        elif self.autoencoder == "DDM":
                            epoch_validation_loss.append(loss_val[-1].detach().cpu().numpy())
                        else:
                            epoch_validation_loss.append(loss_val.detach().cpu().numpy())
                    
                        
                self.autoencoding_losses_validation.append(epoch_validation_loss)

                save_dict = {'epoch': epoch,
                        'gen_state_dict': self.gen.state_dict(),
                        'pred_state_dict': self.pred.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.autoencoding_losses,
                        'validation_loss': self.autoencoding_losses_validation
                        }
                
                if self.autoencoder == 'DDM':
                    save_dict['dyn_state_dict'] = self.dyn.state_dict()
                    
                try:
                    torch.save(save_dict, self.checkpoint_file)
                    # torch.save(save_dict, self.checkpoint_file[:-3] + str(epoch) +'_.pt')
                    np.save(self.data_folder + '/{}_losses.npy'.format(self.autoencoder.lower()), np.array(self.autoencoding_losses))
                except Exception as e:
                    print(e)

                
                try:
                    np.save(self.data_folder + '/{}_validation_losses.npy'.format(self.autoencoder.lower()), np.array(self.autoencoding_losses_validation))
                except Exception as e:
                    print(e)
                    
            #Final epoch checkpoint
            try:
                save_dict = {
                            'epoch': self.autoencoder_num_epochs-1,
                            'gen_state_dict': self.gen.state_dict(),
                            'pred_state_dict': self.pred.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.autoencoding_losses,
                            'validation_loss': self.autoencoding_losses_validation,
                            }
                if self.autoencoder == 'DDM':
                    save_dict['dyn_state_dict'] = self.dyn.state_dict()
                    torch.save(self.dyn.state_dict(), self.dyn_file)
                torch.save(self.gen.state_dict(), self.gen_file)
                torch.save(self.pred.state_dict(), self.pred_file)
                torch.save(save_dict, self.checkpoint_file)
                np.save(self.data_folder + '/{}_losses.npy'.format(self.autoencoder.lower()), np.array(self.autoencoding_losses))
            except Exception as e:
                    print(e)
           
        
    def evaluate_trained_model(self):
        '''After training, this method can be called to use the trained autoencoder to embed all the data in the representation space.
        We encode all data subsets (train, validation and test) separately and save them off as independent tuples. We then will
        also combine these subsets to populate a replay buffer to train a policy from.
        
        This method will also evaluate the decoder's ability to correctly predict the next observation from the and also will
        evaluate the trained representation's correlation with the acuity scores.
        '''

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, encoded_state=True, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))

        errors = []
        correlations = torch.Tensor()
        test_representations = torch.Tensor()
        print('Encoding the Training and Validataion Data.')
        ## LOOP THROUGH THE DATA
        # -----------------------------------------------
        # For Training and Validation sets (Encode the observations only, add all data to the experience replay buffer)
        # For the Test set:
        # - Encode the observations
        # - Save off the data (as test tuples and place in the experience replay buffer)
        # - Evaluate accuracy of predicting the next observation using the decoder module of the model
        # - Evaluate the correlation coefficient between the learned representations and the acuity scores
        with torch.no_grad():
            for i_set, loader in enumerate([self.train_loader, self.val_loader, self.test_loader]):
                if i_set == 2:
                    print('Encoding the Test Data. Evaluating prediction accuracy. Calculating Correlation Coefficients.')
                for dem, ob, ac, l, t, scores, rewards, idx in loader:
                    dem = dem.to(self.device)
                    ob = ob.to(self.device)
                    ac = ac.to(self.device)
                    l = l.to(self.device)
                    t = t.to(self.device)
                    scores = scores.to(self.device)
                    rewards = rewards.to(self.device)

                    max_length = int(l.max().item())

                    ob = ob[:,:max_length,:]
                    dem = dem[:,:max_length,:]
                    ac = ac[:,:max_length,:]
                    scores = scores[:,:max_length,:]
                    rewards = rewards[:,:max_length]

                    cur_obs, next_obs = ob[:,:-1,:], ob[:,1:,:]
                    cur_dem, next_dem = dem[:,:-1,:], dem[:,1:,:]
                    cur_actions = ac[:,:-1,:]
                    cur_rewards = rewards[:,:-1]
                    cur_scores = scores[:,:-1,:]
                    mask = (cur_obs==0).all(dim=2)
                    
                    self.gen.eval()
                    self.pred.eval()

                    if self.autoencoder in ['AE', 'AIS', 'RNN']:

                        if self.context_input:
                            representations = self.gen(torch.cat((cur_obs, cur_dem, torch.cat((torch.zeros((ob.shape[0],1,ac.shape[-1])).to(self.device),ac[:,:-2,:]),dim=1)),dim=-1))
                        else:
                            representations = self.gen(torch.cat((cur_obs, torch.cat((torch.zeros((ob.shape[0],1,ac.shape[-1])).to(self.device), ac[:,:-2,:]),dim=1)), dim=-1))

                        if self.autoencoder == 'RNN':
                            pred_obs = self.pred(representations)
                        else:
                            pred_obs = self.pred(torch.cat((representations,cur_actions),dim=-1))

                        pred_error = F.mse_loss(next_obs[~mask], pred_obs[~mask])

                    elif self.autoencoder == 'DDM':
                        # Initialize hidden states for the LSTM layer
                        cx_d = torch.zeros(1, ob.shape[0], self.hidden_size).to(self.device)
                        hx_d = torch.zeros(1, ob.shape[0], self.hidden_size).to(self.device)

                        if self.context_input:
                            representations = self.gen(torch.cat((cur_obs, cur_dem), dim=-1))
                            z_prime = self.gen(torch.cat((next_obs, next_dem), dim=-1))
                        else:
                            representations = self.gen(cur_obs)
                            z_prime = self.gen(next_obs)

                        s_hat = self.pred(representations)
                        z_prime_hat, a_hat, _ = self.dyn((representations, z_prime, cur_actions, (hx_d, cx_d)))
                        s_prime_hat = self.pred(z_prime_hat)

                        __, pred_error, __, __, __ = get_dynamics_losses(
                            cur_obs[~mask], s_hat[~mask], next_obs[~mask], s_prime_hat[~mask], z_prime[~mask], z_prime_hat[~mask],
                            a_hat[~mask], cur_actions[~mask], discrete=False)                  
                    
                    elif self.autoencoder in ['DST', 'ODERNN']:
                        _, pred_error, representations = self.container.loop(ob, dem, ac, scores, l, max_length, self.context_input, corr_coeff_param = 0, device=self.device)
                        representations = representations[:, :-1, :].detach() # remove latent of last time step (with no target) 
                        
                    elif self.autoencoder == 'CDE':
                        i_coefs = (self.train_coefs, self.val_coefs, self.test_coefs)[i_set]                         
                        _, pred_error, representations = self.container.loop(ob, dem, ac, scores, l, max_length, self.context_input, corr_coeff_param = 0, device=self.device, coefs = i_coefs, idx = idx) 
                        representations = representations[:, :-1, :].detach()
                                                

                    if i_set == 2:  # If we're evaluating the models on the test set...
                        # Compute the Pearson correlation of the learned representations and the acuity scores
                        corr = torch.zeros((cur_obs.shape[0], representations.shape[-1], cur_scores.shape[-1]))
                        for i in range(cur_obs.shape[0]):
                            corr[i] = pearson_correlation(representations[i][~mask[i]], cur_scores[i][~mask[i]], device=self.device)
                
                        # Concatenate this batch's correlations with the larger tensor
                        correlations = torch.cat((correlations, corr), dim=0)

                        # Concatenate the batch's representations with the larger tensor
                        test_representations = torch.cat((test_representations, representations.cpu()), dim=0)

                        # Append the batch's prediction errors to the list
                        if torch.is_tensor(pred_error):
                            errors.append(pred_error.item())
                        else:
                            errors.append(pred_error)

                    # Remove values with the computed mask and add data to the experience replay buffer
                    cur_rep = torch.cat((representations[:,:-1, :], torch.zeros((cur_obs.shape[0], 1, self.hidden_size)).to(self.device)), dim=1)
                    next_rep = torch.cat((representations[:,1:, :], torch.zeros((cur_obs.shape[0], 1, self.hidden_size)).to(self.device)), dim=1)
                    cur_rep = cur_rep[~mask].cpu()
                    next_rep = next_rep[~mask].cpu()
                    cur_actions = cur_actions[~mask].cpu()
                    cur_rewards = cur_rewards[~mask].cpu()
                    cur_obs = cur_obs[~mask].cpu()  # Need to keep track of the actual observations that were made to form the corresponding representations (for downstream WIS)
                    next_obs = next_obs[~mask].cpu()
                    cur_dem = cur_dem[~mask].cpu()
                    next_dem = next_dem[~mask].cpu()
                    
                    # Loop over all transitions and add them to the replay buffer
                    for i_trans in range(cur_rep.shape[0]):
                        done = cur_rewards[i_trans] != 0
                        if self.context_input:
                            self.replay_buffer.add(cur_rep[i_trans].numpy(), cur_actions[i_trans].argmax().item(), next_rep[i_trans].numpy(), cur_rewards[i_trans].item(), done.item(), torch.cat((cur_obs[i_trans],cur_dem[i_trans]),dim=-1).numpy(), torch.cat((next_obs[i_trans], next_dem[i_trans]), dim=-1).numpy())
                        else:
                            self.replay_buffer.add(cur_rep[i_trans].numpy(), cur_actions[i_trans].argmax().item(), next_rep[i_trans].numpy(), cur_rewards[i_trans].item(), done.item(), cur_obs[i_trans].numpy(), next_obs[i_trans].numpy())

            ## SAVE OFF DATA
            # --------------
            self.replay_buffer.save(self.buffer_save_file)
            torch.save(errors, self.next_obs_pred_errors_file)
            torch.save(test_representations, self.test_representations_file)
            torch.save(correlations, self.test_correlations_file)
  
  
    def train_dBCQ_policy(self, pol_learning_rate=1e-3):

        # Initialize parameters for policy learning
        params = {
            "eval_freq": 500,
            "discount": 0.99,
            "buffer_size": 350000,
            "batch_size": self.minibatch_size,
            "optimizer": "Adam",
            "optimizer_parameters": {
                "lr": pol_learning_rate
            },
            "train_freq": 1,
            "polyak_target_update": True,
            "target_update_freq": 1,
            "tau": 0.01,
            "max_timesteps": 5e5,
            "BCQ_threshold": 0.3,
            "buffer_dir": self.buffer_save_file,
            "policy_file": self.policy_save_file+f'_l{pol_learning_rate}.pt',
            "pol_eval_file": self.policy_eval_save_file+f'_l{pol_learning_rate}.npy',
        }
        
        # Initialize a dataloader for policy evaluation (will need representations, observations, demographics, rewards and actions from the test dataset)
        test_representations = torch.load(self.test_representations_file)  # Load the test representations
        pol_eval_dataset = TensorDataset(test_representations, self.test_states, self.test_interventions, self.test_demog, self.test_rewards)
        pol_eval_dataloader = DataLoader(pol_eval_dataset, batch_size=self.minibatch_size, shuffle=False)

        # Initialize and Load the experience replay buffer corresponding with the current settings of rand_num, hidden_size, etc...
        replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, encoded_state=True, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))

        # Load the pretrained policy for whether or not the demographic context was used to train the representations 
        behav_input = self.state_dim + (self.context_dim if self.context_input else 0)
        behav_pol = FC_BC(behav_input, self.num_actions, 64).to(self.device)
        if self.context_input:
            behav_pol.load_state_dict(torch.load(self.behav_policy_file_wDemo))
        else:
            behav_pol.load_state_dict(torch.load(self.behav_policy_file))
        behav_pol.eval()

        # Run dBCQ_utils.train_dBCQ
        train_dBCQ(replay_buffer, self.num_actions, self.hidden_size, self.device, params, behav_pol, pol_eval_dataloader, self.context_input)

