'''
This script preprocesses and organizes the Sepsis patient cohort extracted with the procedure 
provided at: https://github.com/microsoft/mimic_sepsis to produce patient trajectories for easier
use in sequential models.

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

'''

import sys
import os
import time
import torch

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

save_dir = 'data/sepsis_mimiciii/'
train_file = 'train_set_tuples'
val_file = 'val_set_tuples'
test_file = 'test_set_tuples'

# This assumes MIMIC-III sepsis cohort has been extracted from https://github.com/microsoft/mimic_sepsis
full_data_file = os.path.join(save_dir, 'sepsis_final_data_withTimes.csv')
acuity_scores_file = os.path.join(save_dir, 'acuity_scores.csv')  # These are extracted using derive_acuities.py

full_zs = pd.read_csv(full_data_file)
acuity_scores = pd.read_csv(acuity_scores_file)

## Determine the train, val, test split (70/15/15), stratified by patient outcome
temp = full_zs.groupby('traj')['r:reward'].sum()
y = temp.values
X = temp.index.values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5)

# Drop unneeded meta features
full_zs = full_zs.drop(['m:presumed_onset', 'm:charttime', 'm:icustayid'], axis=1)

train_data = full_zs[full_zs['traj'].isin(X_train)]
train_acuity = acuity_scores[acuity_scores['traj'].isin(X_train)]
trajectories = train_data['traj'].unique()

val_data = full_zs[full_zs['traj'].isin(X_val)]
val_acuity = acuity_scores[acuity_scores['traj'].isin(X_val)]
val_trajectories = val_data['traj'].unique()

test_data = full_zs[full_zs['traj'].isin(X_test)]
test_acuity = acuity_scores[acuity_scores['traj'].isin(X_test)]
test_trajectories = test_data['traj'].unique()

# Define the features of the full data
num_actions = 25
state_dim = 47
num_obs = 33
num_dem = 5
num_acuity_scores = 3
horizon = 21
device = 'cpu'


################################################################
#          FORMAT DATA FOR USE IN SEQUENTIAL MODELS
################################################################


dem_keep_cols = ['o:gender', 'o:mechvent', 'o:re_admission', 'o:age', 'o:Weight_kg']
obs_keep_cols = ['o:GCS', 'o:HR', 'o:SysBP',
       'o:MeanBP', 'o:DiaBP', 'o:RR', 'o:Temp_C', 'o:FiO2_1', 'o:Potassium',
       'o:Sodium', 'o:Chloride', 'o:Glucose', 'o:Magnesium', 'o:Calcium',
       'o:Hb', 'o:WBC_count', 'o:Platelets_count', 'o:PTT', 'o:PT',
       'o:Arterial_pH', 'o:paO2', 'o:paCO2', 'o:Arterial_BE', 'o:HCO3',
       'o:Arterial_lactate','o:PaO2_FiO2', 'o:SpO2', 'o:BUN', 'o:Creatinine',
       'o:SGOT', 'o:SGPT', 'o:Total_bili', 'o:INR']

dem_cols = [i for i in train_data.columns if i in dem_keep_cols]
obs_cols = [i for i in train_data.columns if i in obs_keep_cols]
ac_cols  = [i for i in train_data.columns if i[:2] == 'a:']
rew_cols = [i for i in train_data.columns if i[:2] == 'r:']
acuity_cols = [i for i in train_acuity.columns if i[:2] == 'c:']
#Assuming discrete actions and scalar rewards:

assert len(obs_cols) > 0, 'No observations present, or observation columns not prefixed with "o:"'
assert len(ac_cols) > 0, 'No actions present, or actions column not prefixed with "a:"'
assert len(rew_cols) > 0, 'No rewards present, or rewards column not prefixed with "r:"'
assert len(ac_cols) == 1, 'Multiple action columns are present when a single action column is expected'
assert len(rew_cols) == 1, 'Multiple reward columns are present when a single reward column is expected'
# assert len(acuity_cols) == num_acuity_scores, 'Ensure that we have the right number of acuity scores'
ac_col = ac_cols[0]
rew_col = rew_cols[0]


## TRAINING DATA
#---------------------------------------------------------------------------
print("Converting Training Data")
print("--"*20)
train_data[ac_col] = train_data[ac_col]
all_actions = train_data[ac_col].unique()
all_actions.sort()
try:
    all_actions = all_actions.astype(np.int32)
except:
    raise ValueError('Actions are expected to be integers, but are not.')
# if not all(all_actions == np.arange(num_actions, dtype=np.int32)):
    # print(Font.red + 'Some actions are missing from data or all action space not properly defined.' + Font.end)
data_trajectory = {}
data_trajectory['dem_cols'] = dem_cols
data_trajectory['obs_cols'] = obs_cols
data_trajectory['ac_col']  = ac_col
data_trajectory['rew_col'] = rew_col
data_trajectory['num_actions'] = num_actions
data_trajectory['obs_dim'] = len(obs_cols)
data_trajectory['traj'] = {}
data_trajectory['pos_traj'] = []
data_trajectory['neg_traj'] = []

for i in trajectories:
    # bar.update()
    traj_i = train_data[train_data['traj'] == i].sort_values(by='step')
    traj_j = train_acuity[train_acuity['traj']==i].sort_values(by='step')
    data_trajectory['traj'][i] = {}
    data_trajectory['traj'][i]['dem'] = torch.Tensor(traj_i[dem_cols].values).to('cpu')
    data_trajectory['traj'][i]['obs'] = torch.Tensor(traj_i[obs_cols].values).to('cpu')
    data_trajectory['traj'][i]['actions'] = torch.Tensor(traj_i[ac_col].values.astype(np.int32)).to('cpu').long()
    data_trajectory['traj'][i]['rewards'] = torch.Tensor(traj_i[rew_col].values).to('cpu')
    data_trajectory['traj'][i]['acuity'] = torch.Tensor(traj_j[acuity_cols].values).to('cpu')
    if sum(traj_i[rew_col].values) > 0:
        data_trajectory['pos_traj'].append(i)
    else:
        data_trajectory['neg_traj'].append(i)

observations = torch.zeros((len(trajectories), horizon, num_obs))
demographics = torch.zeros((len(trajectories), horizon, num_dem)) 
actions = torch.zeros((len(trajectories), horizon-1, num_actions))
lengths = torch.zeros((len(trajectories)), dtype=torch.int)
times = torch.zeros((len(trajectories), horizon))
rewards = torch.zeros((len(trajectories), horizon))
acuities = torch.zeros((len(trajectories), horizon-1, num_acuity_scores))
action_temp = torch.eye(25)
for ii, traj in enumerate(trajectories):
    obs = data_trajectory['traj'][traj]['obs']
    dem = data_trajectory['traj'][traj]['dem']
    action = data_trajectory['traj'][traj]['actions'].view(-1,1)
    reward = data_trajectory['traj'][traj]['rewards']
    acuity = data_trajectory['traj'][traj]['acuity']
    length = obs.shape[0]
    lengths[ii] = length
    temp = action_temp[action].squeeze(1)
    observations[ii] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
    demographics[ii] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
    actions[ii] = torch.cat((temp, torch.zeros((horizon-length-1, 25), dtype=torch.float)))
    times[ii] = torch.Tensor(range(horizon))
    rewards[ii] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))
    acuities[ii] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))

# Eliminate single transition trajectories...
actions = actions[lengths>1.0].to(device)
observations = observations[lengths>1.0].to(device)
demographics = demographics[lengths>1.0].to(device)
times = times[lengths>1.0].to(device)
rewards = rewards[lengths>1.0].to(device)
acuities = acuities[lengths>1.0].to(device)
lengths = lengths[lengths>1.0].to(device)


## Validation DATA
#---------------------------------------------------------------------------
print("Converting Validation Data")
print("="*20)
val_data_trajectory = {}
val_data_trajectory['obs_cols'] = obs_cols
val_data_trajectory['dem_cols'] = dem_cols
val_data_trajectory['ac_col']  = ac_col
val_data_trajectory['rew_col'] = rew_col
val_data_trajectory['num_actions'] = num_actions
val_data_trajectory['obs_dim'] = len(obs_cols)
val_data_trajectory['traj'] = {}
val_data_trajectory['pos_traj'] = []
val_data_trajectory['neg_traj'] = []

for j in val_trajectories:
    traj_j = val_data[val_data['traj']==j].sort_values(by='step')
    traj_k = val_acuity[val_acuity['traj']==j].sort_values(by='step')
    val_data_trajectory['traj'][j] = {}
    val_data_trajectory['traj'][j]['dem'] = torch.Tensor(traj_j[dem_cols].values).to('cpu')
    val_data_trajectory['traj'][j]['obs'] = torch.Tensor(traj_j[obs_cols].values).to('cpu')
    val_data_trajectory['traj'][j]['actions'] = torch.Tensor(traj_j[ac_col].values.astype(np.int32)).to('cpu').long()
    val_data_trajectory['traj'][j]['rewards'] = torch.Tensor(traj_j[rew_col].values).to('cpu')
    val_data_trajectory['traj'][j]['acuity'] = torch.Tensor(traj_k[acuity_cols].values).to('cpu')
    if sum(traj_j[rew_col].values) > 0:
        val_data_trajectory['pos_traj'].append(j)
    else:
        val_data_trajectory['neg_traj'].append(j)

val_obs = torch.zeros((len(val_trajectories), horizon, num_obs))
val_dem = torch.zeros((len(val_trajectories), horizon, num_dem))
val_actions = torch.zeros((len(val_trajectories), horizon-1, num_actions))
val_lengths = torch.zeros((len(val_trajectories)), dtype=torch.int)
val_times = torch.zeros((len(val_trajectories),horizon))
val_rewards = torch.zeros((len(val_trajectories), horizon))
val_acuities = torch.zeros((len(val_trajectories), horizon-1, num_acuity_scores))
action_temp = torch.eye(25)
for jj, traj in enumerate(val_trajectories):
    obs = val_data_trajectory['traj'][traj]['obs']
    dem = val_data_trajectory['traj'][traj]['dem']
    action = val_data_trajectory['traj'][traj]['actions'].view(-1,1)
    reward = val_data_trajectory['traj'][traj]['rewards']
    acuity = val_data_trajectory['traj'][traj]['acuity']
    length = obs.shape[0]
    val_lengths[jj] = length
    temp = action_temp[action].squeeze(1)
    val_obs[jj] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
    val_dem[jj] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
    val_actions[jj] = torch.cat((temp, torch.zeros((horizon-length-1, 25), dtype=torch.float)))
    val_times[jj] = torch.Tensor(range(horizon))
    val_rewards[jj] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))
    val_acuities[jj] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))

# Eliminate single transition trajectories...
val_actions = val_actions[val_lengths>1.0].to(device)
val_obs = val_obs[val_lengths>1.0].to(device)
val_dem = val_dem[val_lengths>1.0].to(device)
val_times = val_times[val_lengths>1.0].to(device)
val_rewards = val_rewards[val_lengths>1.0].to(device)
val_acuities = val_acuities[val_lengths>1.0].to(device)
val_lengths = val_lengths[val_lengths>1.0].to(device)


## Test DATA
#---------------------------------------------------------------------------
print("Converting Test Data")
print("+"*20)
test_data_trajectory = {}
test_data_trajectory['obs_cols'] = obs_cols
test_data_trajectory['dem_cols'] = dem_cols
test_data_trajectory['ac_col']  = ac_col
test_data_trajectory['rew_col'] = rew_col
test_data_trajectory['num_actions'] = num_actions
test_data_trajectory['obs_dim'] = len(obs_cols)
test_data_trajectory['traj'] = {}
test_data_trajectory['pos_traj'] = []
test_data_trajectory['neg_traj'] = []

for j in test_trajectories:
    traj_j = test_data[test_data['traj']==j].sort_values(by='step')
    traj_k = test_acuity[test_acuity['traj']==j].sort_values(by='step')
    test_data_trajectory['traj'][j] = {}
    test_data_trajectory['traj'][j]['obs'] = torch.Tensor(traj_j[obs_cols].values).to('cpu')
    test_data_trajectory['traj'][j]['dem'] = torch.Tensor(traj_j[dem_cols].values).to('cpu')
    test_data_trajectory['traj'][j]['actions'] = torch.Tensor(traj_j[ac_col].values.astype(np.int32)).to('cpu').long()
    test_data_trajectory['traj'][j]['rewards'] = torch.Tensor(traj_j[rew_col].values).to('cpu')
    test_data_trajectory['traj'][j]['acuity'] = torch.Tensor(traj_k[acuity_cols].values).to('cpu')
    if sum(traj_j[rew_col].values) > 0:
        test_data_trajectory['pos_traj'].append(j)
    else:
        test_data_trajectory['neg_traj'].append(j)

test_obs = torch.zeros((len(test_trajectories), horizon, num_obs))
test_dem = torch.zeros((len(test_trajectories), horizon, num_dem))
test_actions = torch.zeros((len(test_trajectories), horizon-1, num_actions))
test_lengths = torch.zeros((len(test_trajectories)), dtype=torch.int)
test_times = torch.zeros((len(test_trajectories), horizon))
test_rewards = torch.zeros((len(test_trajectories), horizon))
test_acuities = torch.zeros((len(test_trajectories), horizon-1, num_acuity_scores))
action_temp = torch.eye(25)
for jj, traj in enumerate(test_trajectories):
    obs = test_data_trajectory['traj'][traj]['obs']
    dem = test_data_trajectory['traj'][traj]['dem']
    action = test_data_trajectory['traj'][traj]['actions'].view(-1,1)
    reward = test_data_trajectory['traj'][traj]['rewards']
    acuity = test_data_trajectory['traj'][traj]['acuity']
    length = obs.shape[0]
    test_lengths[jj] = length
    temp = action_temp[action].squeeze(1)
    test_obs[jj] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
    test_dem[jj] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
    test_actions[jj] = torch.cat((temp, torch.zeros((horizon-length-1, 25), dtype=torch.float)))
    test_times[jj] = torch.Tensor(range(horizon))
    test_rewards[jj] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))
    test_acuities[jj] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))

# Eliminate single transition trajectories...
test_actions = test_actions[test_lengths>1.0].to(device)
test_obs = test_obs[test_lengths>1.0].to(device)
test_dem = test_dem[test_lengths>1.0].to(device)
test_times = test_times[test_lengths>1.0].to(device)
test_acuities = test_acuities[test_lengths>1.0].to(device)
#test_mortality = test_mortality[test_lengths>1.0].to(device)
test_rewards = test_rewards[test_lengths>1.0].to(device)
test_lengths = test_lengths[test_lengths>1.0].to(device)


#### Save off the tuples...
#############################
print("Saving off tuples")
print("..."*20)
torch.save((demographics,observations,actions,lengths,times,acuities,rewards),os.path.join(save_dir,train_file))

torch.save((val_dem,val_obs,val_actions,val_lengths,val_times,val_acuities,val_rewards),os.path.join(save_dir,val_file))

torch.save((test_dem,test_obs,test_actions,test_lengths,test_times,test_acuities,test_rewards),os.path.join(save_dir,test_file))

print("\n")
print("Finished conversion")

# We also extract and save off the mortality outcome of patients in the test set for evaluation and analysis purposes
print("\n")
print("Extracting Test set mortality")
test_mortality = torch.Tensor(test_data.groupby('traj')['r:reward'].sum().values)
test_mortality = test_mortality.unsqueeze(1).unsqueeze(1)
test_mortality = test_mortality.repeat(1,20,1)  # Put in the same general format as the patient trajectories
# Save off mortality tuple
torch.save(test_mortality,os.path.join(save_dir,'test_mortality_tuple'))
