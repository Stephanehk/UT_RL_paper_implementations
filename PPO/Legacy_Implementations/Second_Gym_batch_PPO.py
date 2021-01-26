import collections
import queue
import numpy as np
import cv2
import argparse
import logging
import time
import math
import random
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
import gym
import copy
from itertools import count

from memory import Memory
from running_state import ZFilter

class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.module_list_current = [self.affine1, self.affine2, self.action_mean, self.action_log_std]

        self.module_list_old = [None]*len(self.module_list_current) #self.affine1_old, self.affine2_old, self.action_mean_old, self.action_log_std_old]
        self.backup()

    def backup(self):
        for i in range(len(self.module_list_current)):
            self.module_list_old[i] = copy.deepcopy(self.module_list_current[i])

    def forward(self, x, old=False):
        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))

            action_mean = self.module_list_old[2](x)
            action_log_std = self.module_list_old[3].expand_as(action_mean)
            action_std = torch.exp(action_log_std)
        else:
            x = F.tanh(self.affine1(x))
            x = F.tanh(self.affine2(x))

            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

env = gym.make("LunarLanderContinuous-v2")

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# env.seed(args.seed)
# torch.manual_seed(args.seed)
gamma = 0.99
tau = 0.95
clip_epsilon = 0.2
batch_size = 5000

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
opt_policy = optim.Adam(policy_net.parameters(), lr=0.005)
opt_value = optim.Adam(value_net.parameters(), lr=0.003)

torch.set_default_tensor_type('torch.FloatTensor')
PI = torch.FloatTensor([3.1415926])

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0).float()
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    opt_value.zero_grad()
    value_loss = (values - targets).pow(2.).mean()
    value_loss.backward()
    opt_value.step()

    action_var = Variable(actions)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old = policy_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    policy_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_policy.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    opt_policy.step()

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []

logging_n_eps = 0
reward_batch = 0
total_eps = 0

for i_episode in range(1000):
    memory = Memory()

    num_steps = 0
    num_episodes = 0
    while num_steps < batch_size:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            #if args.render:
            env.render()
            if done:
                break

            state = next_state

        num_steps += (t-1)
        num_episodes += 1
        total_eps += 1
        logging_n_eps += 1
        reward_batch += reward_sum

        if total_eps % 20 == 0:
            print ("Avg batch return after " + str(total_eps) + " episodes: " + str(reward_batch/logging_n_eps))
            logging_n_eps = 0
            reward_batch = 0


    batch = memory.sample()

    update_params(batch)
