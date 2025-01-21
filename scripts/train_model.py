'''
This script configures and executes experiments for evaluating recurrent autoencoding approaches useful for learning
informative representations of sequentially observed patient health.

After configuring the specific settings and hyperparameters for the selected autoencoder, the experiment can be specified to:
(1) Train the selected encoding and decoding functions used to establish the learned state representations 
(2) Evaluate the trained model and encode+save the patient trajectories by their learned representations
(3) Learn a treatment policy using the saved patient representations via offline RL. The algorithm used to learn a policy
    is the discretized form of Batch Constrained Q-learning [Fujimoto, et al (2019)]

The patient cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:
 - The code for the AIS approach and general framework we build from was developed by Jayakumar Subramanian

'''

import random
import os
import sys
import pickle
import click
import yaml
import numpy as np
from experiment import Experiment

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=2)

@click.command()
@click.option('--autoencoder', '-a', type=click.Choice(['AE', 'AIS', 'CDE', 'DDM', 'DST', 'ODERNN', 'RNN']))
@click.option('--domain', '-d', default='sepsis', help="Only 'sepsis' implemented for now")
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(autoencoder, domain, options):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = yaml.safe_load(open(os.path.join(dir_path, '../configs/common.yaml'), 'r'))    
    cfg_file = os.path.join(dir_path, '../configs/config_' + domain + f'_{autoencoder.lower()}.yaml')
    model_params = yaml.safe_load(open(cfg_file, 'r'))
    
    if autoencoder == 'CDE':
        model_params['coefs_folder'] =  os.path.join(params['storage_path'], model_params['coefs_folder'])
            
    # Iterating over the keys in model_params and replacing the values in params merging the two dictionaries
    for i in model_params:
        params[i] = model_params[i]        

    # Overriding params from config file with command line options if provided
    for opt in options:
        print(opt)
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    # Printing final parameters
    print('Parameters ')
    for key in params:
        print(key, params[key])
    print('=' * 30)

    
    if params['device'] == 'cuda':
        if torch.cuda.is_available():
            params['device'] = torch.device('cuda')
        else:
            params['device'] = torch.device('cpu')

    random_seed = params['random_seed']
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random_state = np.random.RandomState(random_seed)
    params['rng'] = random_state
    params['domain'] = domain
        
    # Update foldername to the full path
    folder_name = params['storage_path'] + params['folder_location'] + params['folder_name']
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    params['folder_name'] = folder_name
    
    torch.set_num_threads(torch.get_num_threads())
    
    # Save model_params as a separate key in params also
    params[f'{autoencoder.lower()}_hypers'] = model_params # Cortex hyperparameter dictionaries 
    
    # Experiment
    experiment = Experiment(**params)    
    experiment.train_autoencoder()
    experiment.evaluate_trained_model()
    experiment.train_dBCQ_policy(params['pol_learning_rate'])
    print('=' * 30)

    with open(folder_name + '/config.yaml', 'w') as y:
        yaml.safe_dump(params, y)  # saving params for reference

if __name__ == '__main__':
    run()