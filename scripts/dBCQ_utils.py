"""
The classes and methods in this file are derived or pulled directly from https://github.com/sfujim/BCQ/tree/master/discrete_BCQ
which is a discrete implementation of BCQ by Scott Fujimoto, et al. and featured in the following 2019 DRL NeurIPS workshop paper:
@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti, Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


# Simple full-connected supervised network for Behavior Cloning of batch data
class FC_BC(nn.Module):
	def __init__(self, state_dim=33, num_actions=25, num_nodes=64):
		super(FC_BC, self).__init__()
		self.l1 = nn.Linear(state_dim, num_nodes)
		self.bn1 = nn.BatchNorm1d(num_nodes)
		self.l2 = nn.Linear(num_nodes, num_nodes)
		self.bn2 = nn.BatchNorm1d(num_nodes)
		self.l3 = nn.Linear(num_nodes, num_actions)

	def forward(self, state):
		out = F.relu(self.l1(state))
		out = self.bn1(out)
		out = F.relu(self.l2(out))
		out = self.bn2(out)
		return self.l3(out)


# Simple fully-connected Q-network for the policy
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions, num_nodes=128):
		super(FC_Q, self).__init__()
		self.q1 = nn.Linear(state_dim, num_nodes)
		self.q2 = nn.Linear(num_nodes, num_nodes)
		self.q3 = nn.Linear(num_nodes, num_actions)

		self.i1 = nn.Linear(state_dim, num_nodes)
		self.i2 = nn.Linear(num_nodes, num_nodes)
		self.i3 = nn.Linear(num_nodes, num_actions)		


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = F.relu(self.i3(i))
		return self.q3(q), F.log_softmax(i, dim=1), i


class BehaviorCloning(object):
	def __init__(self, input_dim, num_actions, num_nodes=256, learning_rate=1e-3, weight_decay=0.1, optimizer_type='adam', device='cpu'):
		'''Implement a fully-connected network that produces a supervised prediction of the actions
		preserved in the collected batch of data following observations of patient health.
		INPUTS:
		input_dim: int, the dimension of an input array. Default: 33
		num_actions: int, the number of actions available to choose from. Default: 25
		num_nodes: int, the number of nodes
		'''

		self.device = device
		self.state_shape = input_dim
		self.num_actions = num_actions
		self.lr = learning_rate

		# Initialize the network
		self.model = FC_BC(input_dim, num_actions, num_nodes).to(self.device)
		self.loss_func = nn.CrossEntropyLoss()
		if optimizer_type == 'adam':		
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
		else:
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)

		self.iterations = 0

	def train_epoch(self, train_dataloader):
		'''Sample batches of data from training dataloader, predict actions using the network,
		Update the parameters of the network using CrossEntropyLoss.'''

		losses = []

		# Loop through the training data
		for state, action in train_dataloader:
			state = state.to(self.device)
			action = action.to(self.device)

			# Predict the action with the network
			pred_actions = self.model(state)

			# Compute loss
			try:
				loss = self.loss_func(pred_actions, action.flatten())
			except:
				print("LOL ERRORS")

			# Optimize the network
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			losses.append(loss.item())

		self.iterations += 1

		return np.mean(losses)


class discrete_BCQ(object):
	def __init__(
		self, 
		num_actions,
		state_dim,
		device,
		BCQ_threshold=0.3,
		discount=0.99,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=False,
		target_update_frequency=1e3,
		tau=0.005
	):
	
		self.device = device

		# Determine network type
		self.Q = FC_Q(state_dim, num_actions).to(self.device)
		self.Q_target = copy.deepcopy(self.Q)
		self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# # Decay for eps
		# self.initial_eps = initial_eps
		# self.end_eps = end_eps
		# self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1, state_dim)
		# self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0

	# NOTE: This function is only usable when doing online evaluation with a simulator.
	# NOTE: This is why the function is commented out along with all epsilon params, we're not using them
	# def select_action(self, state, eval=False):
	# 	# Select action according to policy with probability (1-eps)
	# 	# otherwise, select random action
	# 	if np.random.uniform(0,1) > self.eval_eps:
	# 		with torch.no_grad():
	# 			state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
	# 			q, imt, i = self.Q(state)
	# 			imt = imt.exp()
	# 			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
	# 			# Use large negative number to mask actions from argmax
	# 			return int((imt * q + (1. - imt) * -1e8).argmax(1))
	# 	else:
	# 		return np.random.randint(self.num_actions)


	def train(self, replay_buffer):
		# Sample replay buffer
		state, action, next_state, reward, done, obs_state, next_obs_state = replay_buffer.sample()

		# Compute the target Q value
		with torch.no_grad():
			q, imt, i = self.Q(next_state)
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

			# Use large negative number to mask actions from argmax
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

			q, imt, i = self.Q_target(next_state)
			target_Q = 10*reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

		# Get current Q estimate
		current_Q, imt, i = self.Q(state)
		current_Q = current_Q.gather(1, action)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q, target_Q)
		i_loss = F.nll_loss(imt, action.reshape(-1))

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

		# Optimize the Q
		self.Q_optimizer.zero_grad()
		Q_loss.backward()
		self.Q_optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


	def polyak_target_update(self):
		for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			self.Q_target.load_state_dict(self.Q.state_dict())


def train_dBCQ(replay_buffer, num_actions, state_dim, device, parameters, behav_pol, pol_eval_dataloader, is_demog):
	# For saving files
	pol_eval_file = parameters['pol_eval_file']
	pol_file = parameters['policy_file']
	buffer_dir = parameters['buffer_dir']

	# Initialize and load policy
	policy = discrete_BCQ(
		num_actions,
		state_dim,
		device,
		parameters["BCQ_threshold"],
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"]
	)

	# Load replay buffer
	replay_buffer.load(buffer_dir, bootstrap=True)

	evaluations = []
	episode_num = 0
	done = True
	training_iters = 0

	while training_iters < parameters["max_timesteps"]:

		for _ in range(int(parameters["eval_freq"])):
			policy.train(replay_buffer)

		evaluations.append(eval_policy(policy, behav_pol, pol_eval_dataloader, parameters["discount"], is_demog, device))  # TODO Run weighted importance sampling with learned policy and behavior policy
		np.save(pol_eval_file, evaluations)
		torch.save({'policy_Q_function':policy.Q.state_dict(), 'policy_Q_target':policy.Q_target.state_dict()}, pol_file)

		training_iters += int(parameters["eval_freq"])
		print(f"Training iterations: {training_iters}")

'''The following is the original dBCQ's evaluation script that we'll need to replace 
with weighted importance sampling between the learned `policy` and the observed policy'''
# # Runs policy for X episodes and returns average reward
# # A fixed seed is used for the eval environment
def eval_policy(policy, behav_policy, pol_dataloader, discount, is_demog, device):

	wis_est = []
	wis_returns = 0
	wis_weighting = 0

	# Loop through the dataloader (representations, observations, actions, demographics, rewards)
	for representations, obs_state, actions, demog, rewards in pol_dataloader:
		representations = representations.to(device)
		obs_state = obs_state.to(device)
		actions = actions.to(device)
		demog = demog.to(device)

		cur_obs, cur_actions = obs_state[:,:-2,:], actions[:,:-1,:].argmax(dim=-1)
		cur_demog, cur_rewards = demog[:,:-2,:], rewards[:,:-2]

		# Mask out the data corresponding to the padded observations
		mask = (cur_obs==0).all(dim=2)

		# Compute the discounted rewards for each trajectory in the minibatch
		discount_array = torch.Tensor(discount**np.arange(cur_rewards.shape[1]))[None,:]
		discounted_rewards = (discount_array * cur_rewards).sum(dim=-1).squeeze()

		# Evaluate the probabilities of the observed action according to the trained policy and the behavior policy
		with torch.no_grad():
			if is_demog:  # Gather the probability from the observed behavior policy
				p_obs = F.softmax(behav_policy(torch.cat((cur_obs.flatten(end_dim=1), cur_demog.flatten(end_dim=1)), dim=-1)), dim=-1).gather(1, cur_actions.flatten()[:,None]).reshape(cur_obs.shape[:2])
			else:
				p_obs = F.softmax(behav_policy(cur_obs.flatten(end_dim=1)), dim=-1).gather(1, cur_actions.flatten()[:,None]).reshape(cur_obs.shape[:2])
			
			q_val, _, _ = policy.Q(representations)  # Compute the Q values of the dBCQ policy
			p_new = F.softmax(q_val, dim=-1).gather(2, cur_actions[:,:,None]).squeeze()  # Gather the probabilities from the trained policy

		# Check for whether there are any zero probabilities in p_obs and replace with small probability since behav_pol may mispredict actual actions...
		if not (p_obs > 0).all(): 
			p_obs[p_obs==0] = 0.1

		# Eliminate spurious probabilities due to padded observations after trajectories have concluded 
		# We do this by forcing the probabilities for these observations to be 1 so they don't affect the product
		p_obs[mask] = 1.
		p_new[mask] = 1.

		cum_ir = torch.clamp((p_new / p_obs).prod(axis=1), 1e-30, 1e4)

		# wis_idx = (cum_ir > 0)  # TODO check that wis_idx isn't empty (all zero)
		# if wis_idx.sum() == 0:
		# 	import pdb; pdb.set_trace()

		# wis = (cum_ir / cum_ir.mean()).cpu() * discounted_rewards  # TODO check that there aren't any nans
		# wis_est.extend(wis.cpu().numpy())
		wis_rewards = cum_ir.cpu() * discounted_rewards
		wis_returns +=  wis_rewards.sum().item()
		wis_weighting += cum_ir.cpu().sum().item()

	wis_eval = (wis_returns / wis_weighting) 
	print("---------------------------------------")
	print(f"Evaluation over the test set: {wis_eval:.3f}")
	print("---------------------------------------")
	return wis_eval