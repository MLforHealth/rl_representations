"""
This script is used to develop a baseline policy using only the observed patient data via Behavior Cloning.

This baseline policy is then used to truncate and guide evaluation of policies learned using dBCQ. It should only need to be
run once for each unique cohort that one looks to learn a better treatment policy for.

The patient cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

# IMPORTS
import argparse
import os
import sys
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dBCQ_utils import BehaviorCloning
from utils import ReplayBuffer


def run(BC_network, train_dataloader, val_dataloader, num_epochs, storage_dir, loss_func, device):
    # Construct training and validation loops
    validation_losses = []
    training_losses = []
    training_iters = 0
    eval_frequency = 100
	
    for i_epoch in range(num_epochs):
        
        train_loss = BC_network.train_epoch(train_dataloader)
        training_losses.append(train_loss)

        if i_epoch % eval_frequency == 0:
            eval_errors = []
            BC_network.model.eval()
            with torch.no_grad():
                for val_state, val_action in val_dataloader:
                    val_state = val_state.to(device)
                    val_action = val_action.to(device)
                    pred_actions = BC_network.model(val_state)
                    try:
                        eval_loss = loss_func(pred_actions, val_action.flatten())
                        eval_errors.append(eval_loss.item())
                    except:
                        print("LOL ERRORS")

            mean_val_loss = np.mean(eval_errors)
            validation_losses.append(mean_val_loss)
            np.save(storage_dir+'validation_losses.npy', validation_losses)
            np.save(storage_dir+'training_losses.npy', training_losses)

            print(f"Training iterations: {i_epoch}, Validation Loss: {mean_val_loss}")
            # Save off and store trained BC model
            torch.save(BC_network.model.state_dict(), storage_dir+'BC_model.pt')

            BC_network.model.train()
    
    print("Finished training Behavior Cloning model")
    print('+='*30)


if __name__ == '__main__':

    # Instead of the commented out code below, we use config files

    # Define input arguments and parameters to override the ones in the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--demographics', dest='dem_context', default=True, action='store_true')
    parser.add_argument('--num_nodes', dest='num_nodes', default=128, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4, type=float)
    parser.add_argument('--storage_folder', dest='storage_folder', default='test/', type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', default=5000, type=int)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.1, type=float)
    parser.add_argument('--optimizer_type', dest='optim_type', default='adam', type=str)

    args = parser.parse_args()

    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(f"{dir_path=}")
    behavCloning_params = yaml.safe_load(open(os.path.join(dir_path, '../configs/config_behavCloning.yaml'), 'r'))
    common_params = yaml.safe_load(open(os.path.join(dir_path, '../configs/common.yaml'), 'r')) 


    # Overriding the parameters in the config file with the input arguments
    behavCloning_params.update(vars(args))

    

    
    # setting device on GPU if available, else CPU
    if common_params['device'] == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    elif common_params['device'] == 'cpu':
        device = torch.device('cpu')
    else:
        print("Please set device to 'cuda' or 'cpu'")
        exit(1)

    print('Using device:', device)
    

    input_dim = 38 if common_params["dem_context"] else 33
    num_actions = 25

    train_buffer_file = behavCloning_params['train_buffer']
    validation_buffer_file = behavCloning_params['val_buffer']

    # Storage folder name is imported from the config file
    storage_dir = behavCloning_params['storage_folder']
    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)

    # Initialize and load the training and validation buffers to populate dataloaders
    train_buffer = ReplayBuffer(input_dim, behavCloning_params["batch_size"], 200000, device)
    train_buffer.load(train_buffer_file)
    states = train_buffer.state[:train_buffer.crt_size]
    actions = train_buffer.action[:train_buffer.crt_size]
    train_dataset = TensorDataset(torch.from_numpy(states).float(), torch.from_numpy(actions).long())
    train_dataloader = DataLoader(train_dataset, batch_size=behavCloning_params["batch_size"], shuffle=True)
    
    val_buffer = ReplayBuffer(input_dim, behavCloning_params["batch_size"], 50000, device)
    val_buffer.load(validation_buffer_file)
    val_states = val_buffer.state[:val_buffer.crt_size]
    val_actions = val_buffer.action[:val_buffer.crt_size]
    val_dataset = TensorDataset(torch.from_numpy(val_states).float(), torch.from_numpy(val_actions).long())
    val_dataloader = DataLoader(val_dataset, batch_size=behavCloning_params["batch_size"], shuffle=False)

    # Initialize the BC network
    BC_network = BehaviorCloning(input_dim, num_actions, behavCloning_params["num_nodes"], behavCloning_params["learning_rate"], behavCloning_params["weight_decay"], behavCloning_params["optim_type"], device)

    loss_func = nn.CrossEntropyLoss()

    run(BC_network, train_dataloader, val_dataloader, behavCloning_params["num_epochs"], storage_dir, loss_func, device)