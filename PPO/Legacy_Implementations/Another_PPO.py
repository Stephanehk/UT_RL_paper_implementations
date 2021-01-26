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
from torch.optim import Adam
import torchvision.transforms as T
from torch.autograd import Variable
import gym
import copy
from itertools import count

from memory import Memory
from running_state import ZFilter

#Source:
#https://github.com/tpbarron/pytorch-ppo

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
grad_clip_param = 40
batch_size = 4000
episodes_per_update = 4
a_lr = 0.005
c_lr = 0.003
n_epochs = 4
batch_size = 128
train_ters = 100
max_steps = 100000

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
opt_policy = Adam(policy_net.parameters(), lr=a_lr)
opt_value = Adam(value_net.parameters(), lr=c_lr)

torch.set_default_tensor_type('torch.FloatTensor')
PI = torch.FloatTensor([3.1415926])

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0).float()
    action_mean, acition_log_std, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)

    a_log_prob = normal_log_density(Variable(action), action_mean, acition_log_std, action_std)
    return action,a_log_prob

def estimate_advantage(memory):
    v_pred = memory.se_state_values
    rewards = memory.se_rewards
    #https://github.com/lilianweng/deep-reinforcement-learning-gym/blob/4fec4876ad28fe83309efd2cdf2a6f4281a5b23c/playground/policies/ppo.py#L173
    T = len(rewards)
    # Compute TD errors
    td_errors = [rewards[t] + gamma * v_pred[t + 1] - v_pred[t] for t in range(T - 1)]
    td_errors += [rewards[T - 1] + gamma * 0.0 - v_pred[T - 1]]  # handle the terminal state.

    assert len(memory.se_actions_log_probs) == len(v_pred) == len(td_errors) == T

    # Estimate advantage backwards.
    advs = []
    adv_so_far = 0.0
    for delta in td_errors[::-1]:
        adv_so_far = delta + gamma * tau * adv_so_far
        advs.append(adv_so_far)
    advs = advs[::-1]
    assert len(advs) == T
    return advs, advs+v_pred

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)

def update_params(memory):
    for epoch in range (n_epochs):
        eps_frames, eps_mes, actions, log_prob_old,state_values, rewards, terminals, advantages, v_targets = memory.reservoir_sample(batch_size)

        eps_frames = torch.squeeze(torch.FloatTensor(eps_frames))
        actions = torch.squeeze(torch.FloatTensor(actions))
        #print (actions.shape)
        log_prob_old = torch.FloatTensor(log_prob_old)
        advantages = torch.FloatTensor(advantages)
        state_values = torch.FloatTensor(state_values)
        v_targets = torch.FloatTensor(v_targets)
        v_targets = Variable(v_targets)
        state_values.requires_grad = True
        v_targets.requires_grad = True

        for i in range (train_ters):
            opt_value.zero_grad()
            value_loss = (state_values - v_targets).pow(2.).mean()
            value_loss.backward()
            opt_value.step()

            action_var = Variable(actions)

            action_means, action_log_stds, action_stds = policy_net(Variable(eps_frames))
            log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

            # backup params after computing probs but before updating new params
            policy_net.backup()

            advantages = (advantages - advantages.mean()) / advantages.std()
            advantages_var = Variable(advantages)

            opt_policy.zero_grad()
            ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
            surr1 = ratio * advantages_var
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(policy_net.parameters(), grad_clip_param)
            opt_policy.step()

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []

memory = Memory()
reward_batch = 0
n_episodes_update = 0
num_episodes_log = 0
total_num_episodes = 0

for i_episode in range(2000):
    while n_episodes_update < episodes_per_update:
        state = env.reset()
        state = running_state(state)

        reward_sum = 0
        for t in range(max_steps): # Don't infinite loop while learning
            action, action_log_prob= select_action(state)
            action = action.data[0].numpy()
            state_value = value_net(Variable(torch.Tensor(state)))

            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            next_state = running_state(next_state)


            memory.add(state, action, action_log_prob, state_value, reward, next_state, done)
            memory.se_add(state, action, action_log_prob, state_value, reward, next_state, done)

            env.render()
            if done:
                break
            state = next_state

        n_episodes_update+=1
        num_episodes_log+=1
        total_num_episodes+=1
        #compute episode advtanages using the single episodes collected data
        advantages,v_targets = estimate_advantage(memory)
        memory.add_advantages(advantages)
        memory.add_targets(v_targets)

        #clear single episodes collected data
        memory.se_clear()

        reward_batch += reward_sum

    if total_num_episodes % 20 == 0:
        reward_batch /= num_episodes_log
        num_episodes_log = 0
        print ("Avg batch reward on " + str(total_num_episodes) + " episode: " + str(reward_batch))
        reward_batch = 0

    n_episodes_update = 0
    update_params(memory)
