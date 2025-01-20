"""
Collection of utilities used for model learning and evaluation.

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
from models.common import mask_from_lengths
from torch.utils.data import DataLoader
import controldiffeq

logger = logging.getLogger(__name__)

def human_evaluation(env, agent, human_trajectories, use_soc_state=True):
    rewards = []
    for ep, trajectory in enumerate(human_trajectories):
        env.reset()
        agent.reset()
        for action in trajectory:
            env.act(action)
        terminal = False
        agent_reward = 0 # NOT  including reward accumulated along human trajectory
        s = env.get_soc_state() if use_soc_state else env.get_pixel_state()
        while not terminal:
            action = agent.get_action(s, evaluate=True)
            pixel_state, r, terminal, soc_state = env.act(action)
            s = soc_state if use_soc_state else pixel_state
            agent_reward += r
        rewards.append(agent_reward)
    return rewards


def plot(data={}, loc="visualization.pdf", x_label="", y_label="", title="", kind='line',
         legend=True, index_col=None, clip=None, moving_average=False):
    # pass
    if all([len(data[key]) > 1 for key in data]):
        if moving_average:
            smoothed_data = {}
            for key in data:
                smooth_scores = [np.mean(data[key][max(0, i - 10):i + 1]) for i in range(len(data[key]))]
                smoothed_data['smoothed_' + key] = smooth_scores
                smoothed_data[key] = data[key]
            data = smoothed_data
        df = pd.DataFrame(data=data)
        ax = df.plot(kind=kind, legend=legend, ylim=clip)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(loc)
        plt.close()

def write_to_csv(data={}, loc="data.csv"):
    if all([len(data[key]) > 1 for key in data]):
        df = pd.DataFrame(data=data)
        df.to_csv(loc)

class Font:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bgblue = '\033[44m'
    bold = '\033[1m'
    underline = '\033[4m'
    end = '\033[0m'

def one_hot(x, num_x, data_type='numpy', device=None):
    if data_type == 'numpy':
        res = np.zeros(num_x)
    elif data_type == 'torch':
        res = torch.zeros(num_x).to(device)
    res[x] = 1.0
    return res

def process_cde_data(loader, context_input, device):
    assert(len(loader) == 1)
    for dem, ob, ac, l, t, scores, rewards, idx in loader: 
        max_length = int(l.max().item())
        ob = ob[:,:max_length,:].to(device)
        dem = dem[:,:max_length,:].to(device)
        ac = ac[:,:max_length,:].to(device)
        scores = scores[:,:max_length,:].to(device)
        l = l.to(device)
        break
    
    ac_shifted = torch.cat((torch.zeros(ac.shape[0], 1, ac.shape[-1]).to(device), ac[:, :-1, :]), dim = 1) # action at t-1
    if context_input:
        obs_data = torch.cat((ob, dem, ac_shifted), dim = -1)
    else:
        obs_data = torch.cat((ob, ac_shifted), dim = -1)
    obs_mask = mask_from_lengths(l, max_length, device).unsqueeze(-1).expand(*obs_data.shape)
    obs_data[~obs_mask] = torch.tensor(float('nan')).to(device)
    times = torch.arange(max_length, device=device).float()    
    
    augmented_X = torch.cat((times.unsqueeze(0).repeat(obs_data.size(0), 1).unsqueeze(-1)
                             , obs_data), dim = 2)
    
    coeffs = controldiffeq.natural_cubic_spline_coeffs(times, augmented_X)
    return coeffs

def load_cde_data(fold, dataset, path, context_input, device):
    fname = os.path.join(path, f'{fold}_{str(context_input)}')
    if os.path.exists(fname):        
        coefs = torch.load(open(fname, 'rb'))
        assert(coefs[0].shape[0] == len(dataset))
        print(f"Loaded {fold} coefs from file")
    else:
        print(f"Encoding {fold} Data")        
        dummy_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        coefs = process_cde_data(dummy_loader, context_input, device)
        torch.save(coefs, open(fname, 'wb'))
    return coefs    

                   

# Generic replay buffer for standard gym tasks (adapted from https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/utils.py)
class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device, encoded_state=False, obs_state_dim=38):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.encoded_state = encoded_state

        self.ptr = 0
        self.crt_size = 0
        # If not encoded state, we put the not encoded data to self.state, instaead of self.obs_state
        if not encoded_state:
            self.state = np.zeros((self.max_size, obs_state_dim))
            
        else:
            self.state = np.zeros((self.max_size, state_dim))

        self.next_state = np.array(self.state)
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        if encoded_state:
            self.obs_state = np.zeros((self.max_size, obs_state_dim))
            self.next_obs_state = np.zeros((self.max_size, obs_state_dim))


    def add(self, state, action, next_state, reward, done, obs_state=None, next_obs_state=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        if self.encoded_state:
            # Saving the not-encoded observations for the state and next_state
            self.obs_state[self.ptr] = obs_state
            self.next_obs_state[self.ptr] = next_obs_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        if not self.encoded_state:
            return(
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
            )
        else:
            return(
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.obs_state[ind]).to(self.device),
                torch.FloatTensor(self.next_obs_state[ind]).to(self.device)
            )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)
        if self.encoded_state:
            np.save(f"{save_folder}_obs_state.npy", self.obs_state[:self.crt_size])
            np.save(f"{save_folder}_next_obs_state.npy", self.next_obs_state[:self.crt_size])

    def load(self, save_folder, size=-1, bootstrap=False):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]
        if self.encoded_state:
            self.obs_state[:self.crt_size] = np.load(f"{save_folder}_obs_state.npy")[:self.crt_size]
            self.next_obs_state[:self.crt_size] = np.load(f"{save_folder}_next_obs_state.npy")[:self.crt_size]

        if bootstrap:
            # Get the indicies of the above arrays that are non-zero
            nonzero_ind = (self.reward !=0)[:,0]
            num_nonzero = sum(nonzero_ind)
            self.state[self.crt_size:(self.crt_size+num_nonzero)] = self.state[nonzero_ind]
            self.action[self.crt_size:(self.crt_size+num_nonzero)] = self.action[nonzero_ind]
            self.next_state[self.crt_size:(self.crt_size+num_nonzero)] = self.next_state[nonzero_ind]
            self.reward[self.crt_size:(self.crt_size+num_nonzero)] = self.reward[nonzero_ind]
            self.not_done[self.crt_size:(self.crt_size+num_nonzero)] = self.not_done[nonzero_ind]
            if self.encoded_state:
                self.obs_state[self.crt_size:(self.crt_size+num_nonzero)] = self.obs_state[nonzero_ind]
                self.next_obs_state[self.crt_size:(self.crt_size+num_nonzero)] = self.next_obs_state[nonzero_ind]
            
            self.crt_size += num_nonzero

            neg_ind = (self.reward < 0)[:,0]
            num_neg = sum(neg_ind)
            self.state[self.crt_size:(self.crt_size+num_neg)] = self.state[neg_ind]
            self.action[self.crt_size:(self.crt_size+num_neg)] = self.action[neg_ind]
            self.next_state[self.crt_size:(self.crt_size+num_neg)] = self.next_state[neg_ind]
            self.reward[self.crt_size:(self.crt_size+num_neg)] = self.reward[neg_ind]
            self.not_done[self.crt_size:(self.crt_size+num_neg)] = self.not_done[neg_ind]
            if self.encoded_state:
                self.obs_state[self.crt_size:(self.crt_size+num_neg)] = self.obs_state[neg_ind]
                self.next_obs_state[self.crt_size:(self.crt_size+num_neg)] = self.next_obs_state[neg_ind]

            self.crt_size += num_neg

        print(f"Replay Buffer loaded with {self.crt_size} elements.")
