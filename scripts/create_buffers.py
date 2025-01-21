import yaml
import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import ReplayBuffer


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


class Buffer(object):
  def __init__(self, hidden_size, state_dim, context_dim, train_data_file, validation_data_file, batch_size, device, dem_context, train_buffer, val_buffer):
        
        self.hidden_size = hidden_size
        self.minibatch_size = batch_size
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.train_data_file = train_data_file
        self.validation_data_file = validation_data_file
        self.context_input = dem_context
        self.train_buffer = train_buffer
        self.val_buffer = val_buffer

        if device == 'cuda':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif device == 'cpu':
            self.device = torch.device('cpu')
        else:
            print("Please set device to 'cuda' or 'cpu'")
            exit(1)
        print('Using device:', self.device)
            

  def create_buffers_and_save(self):
        replay_buffer_train = ReplayBuffer(self.hidden_size, self.minibatch_size, 200000, self.device, encoded_state=False, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))
        replay_buffer_val = ReplayBuffer(self.hidden_size, self.minibatch_size, 50000, self.device, encoded_state=False, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))
        
        self.train_demog, self.train_states, self.train_interventions, self.train_lengths, self.train_times, self.acuities, self.rewards = torch.load(self.train_data_file)
        train_idx = torch.arange(self.train_demog.shape[0])
        self.train_dataset = TensorDataset(self.train_demog, self.train_states, self.train_interventions,self.train_lengths,self.train_times, self.acuities, self.rewards, train_idx)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.minibatch_size, shuffle=True)

        self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards = torch.load(self.validation_data_file)
        val_idx = torch.arange(self.val_demog.shape[0])
        self.val_dataset = TensorDataset(self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards, val_idx)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.minibatch_size, shuffle=False)

        
        all_loaders_list = [self.train_loader, self.val_loader]
        all_buffers = [replay_buffer_train, replay_buffer_val]
        all_buffers_save_files = [self.train_buffer, self.val_buffer]

        for i_set, loader in enumerate(all_loaders_list):
            print("Creating buffer for set ", i_set)
            replay_buffer = all_buffers[i_set]
            buffer_save_file = all_buffers_save_files[i_set]

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

                mask = (cur_obs==0).all(dim=2)

                cur_actions = cur_actions[~mask].cpu()
                cur_rewards = cur_rewards[~mask].cpu()
                cur_obs = cur_obs[~mask].cpu()
                next_obs = next_obs[~mask].cpu()
                cur_dem = cur_dem[~mask].cpu()
                next_dem = next_dem[~mask].cpu()

               
                # Loop over all transitions and add them to the replay buffer
                for i_trans in range(cur_obs.shape[0]):

                    done = cur_rewards[i_trans] != 0
                    
                    if self.context_input:
                        replay_buffer.add(torch.cat((cur_obs[i_trans],cur_dem[i_trans]),dim=-1).numpy(), cur_actions[i_trans].argmax().item(), torch.cat((next_obs[i_trans], next_dem[i_trans]), dim=-1).numpy(), cur_rewards[i_trans].item(), done.item())
                    else:
                        replay_buffer.add(cur_obs[i_trans].numpy(), cur_actions[i_trans].argmax().item(), next_obs[i_trans].numpy(), cur_rewards[i_trans].item(), done.item())


            # Save the replay buffer
            print("Saving buffer to ", buffer_save_file)
            replay_buffer.save(buffer_save_file)
           



if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(f"{dir_path=}")
    behavCloning_params = yaml.safe_load(open(os.path.join(dir_path, '../configs/config_behavCloning.yaml'), 'r'))
    common_params = yaml.safe_load(open(os.path.join(dir_path, '../configs/common.yaml'), 'r')) 

    torch.manual_seed(common_params['random_seed'])

    # Create folder if it does not exist
    if not os.path.exists(behavCloning_params['train_buffer']):
        os.makedirs(behavCloning_params['train_buffer'])

    if not os.path.exists(behavCloning_params['val_buffer']):
        os.makedirs(behavCloning_params['val_buffer'])

    buffer = Buffer(
        hidden_size = behavCloning_params['hidden_size'],
        batch_size = behavCloning_params['batch_size'],
        state_dim = common_params['state_dim'],
        context_dim = common_params['context_dim'],
        train_data_file = common_params['train_data_file'],
        validation_data_file = common_params['validation_data_file'],
        dem_context = common_params['dem_context'],
        train_buffer = behavCloning_params['train_buffer'],
        val_buffer = behavCloning_params['val_buffer'],
        device = common_params['device'], 
    )
    
    buffer.create_buffers_and_save()

