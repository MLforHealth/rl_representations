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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dBCQ_utils import BehaviorCloning
from utils import ReplayBuffer


def run(BC_network, train_dataloader, val_dataloader, num_epochs, storage_dir, loss_func):
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
                    val_state = val_state.to(torch.device('cuda'))
                    val_action = val_action.to(torch.device('cuda'))
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

    # Define input arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--demographics', dest='dem_context', default=False, action='store_true')
    parser.add_argument('--num_nodes', dest='num_nodes', default=128, type=int)
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4, type=float)
    parser.add_argument('--storage_folder', dest='storage_folder', default='test', type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', default=5000, type=int)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.1, type=float)
    parser.add_argument('--optimizer_type', dest='optim_type', default='adam', type=str)

    args = parser.parse_args()

    device = torch.device('cuda')

    input_dim = 38 if args.dem_context else 33
    num_actions = 25
    if args.dem_context:
        train_buffer_file = '/scratch/ssd001/home/tkillian/ml4h2020_srl/raw_data_buffers/train_buffer' 
        validation_buffer_file = '/scratch/ssd001/home/tkillian/ml4h2020_srl/raw_data_buffers/val_buffer'
    else:
        train_buffer_file = '/scratch/ssd001/home/tkillian/ml4h2020_srl/raw_data_buffers/train_noCntxt_buffer' 
        validation_buffer_file = '/scratch/ssd001/home/tkillian/ml4h2020_srl/raw_data_buffers/val_noCntxt_buffer'

    storage_dir = '/scratch/ssd001/home/tkillian/ml4h2020_srl/BehavCloning/' + args.storage_folder + '/'

    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)

    # Initialize and load the training and validation buffers to populate dataloaders
    train_buffer = ReplayBuffer(input_dim, args.batch_size, 200000, device)
    train_buffer.load(train_buffer_file)
    states = train_buffer.state[:train_buffer.crt_size]
    actions = train_buffer.action[:train_buffer.crt_size]
    train_dataset = TensorDataset(torch.from_numpy(states).float(), torch.from_numpy(actions).long())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_buffer = ReplayBuffer(input_dim, args.batch_size, 50000, device)
    val_buffer.load(validation_buffer_file)
    val_states = val_buffer.state[:val_buffer.crt_size]
    val_actions = val_buffer.action[:val_buffer.crt_size]
    val_dataset = TensorDataset(torch.from_numpy(val_states).float(), torch.from_numpy(val_actions).long())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the BC network
    BC_network = BehaviorCloning(input_dim, num_actions, args.num_nodes, args.learning_rate, args.weight_decay, args.optim_type, device)

    loss_func = nn.CrossEntropyLoss()

    run(BC_network, train_dataloader, val_dataloader, args.num_epochs, storage_dir, loss_func)