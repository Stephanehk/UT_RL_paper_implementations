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
import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import copy

device = torch.device(f"cuda:{args.client_gpu}" if torch.cuda.is_available() else "cpu")

class PPO_Agent(nn.Module):
    def __init__(self, linear_state_dim, action_dim, action_std,lr, gamma, n_epochs,clip_val,device):
        super(PPO_Agent, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(linear_state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(linear_state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.device = device
        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.clip_val = clip_val

    def choose_action(self,state):
        state = torch.FloatTensor(state.reshape(1, -1))
        mean = self.actor(state)
        cov_matrix = torch.diag(self.action_var)

        gauss_dist = MultivariateNormal(mean,cov_matrix)
        action = gauss_dist.sample()
        action_log_prob = gauss_dist.log_prob(action)
        return action, action_log_prob

    def get_training_params(self, state, action):
        state = torch.squeeze(torch.stack(state))
        action = torch.squeeze(torch.stack(action))

        mean = self.actor(state)
        action_expanded = self.action_var.expand_as(mean)
        cov_matrix = torch.diag_embed(action_expanded)

        gauss_dist = MultivariateNormal(mean,cov_matrix)
        action_log_prob = gauss_dist.log_prob(action)
        entropy = gauss_dist.entropy()
        state_value = torch.squeeze(self.critic(state))
        return action_log_prob, state_value, entropy


    def format_state (self,state):
        return torch.FloatTensor(state.reshape(1, -1))

    def discount_rewards(self,r, gamma, terminals):
        """ take 1D float array of rewards and compute discounted reward """
        # from https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/helpers.py
        r = np.array(r)
        rev_terminals = terminals[::-1]
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if rev_terminals[t]:
                running_add = 0
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r.tolist()

    def train(self,memory,prev_policy,iters):
        returns = self.discount_rewards(memory.rewards, self.gamma, memory.terminals)
        returns = torch.tensor(returns).to(self.device)
        actions_log_probs = torch.FloatTensor(memory.actions_log_probs).to(self.device)

        #train PPO
        for i in range(self.n_epochs):
            current_action_log_probs, state_values, entropies = self.get_training_params(memory.eps_frames, memory.actions)
            policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())
            advantage = returns - state_values.detach()
            advantage = (advantage - advantage.mean()) / advantage.std()
            adv_l_update1 = policy_ratio*advantage.float()
            adv_l_update2 = (torch.clamp(policy_ratio, 1-self.clip_val, 1+self.clip_val) * advantage).float()
            adv_l = torch.min(adv_l_update1, adv_l_update2)
            loss_v = self.mse(state_values.float(), returns.float())

            loss = \
                - adv_l \
                + (0.5 * loss_v) \
                - (0.01 * entropies)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # if i % 10 == 0:
            #     print("    on epoch " + str(i))

            # if iters % 50 == 0:
            #     torch.save(self.state_dict(), "vanilla_policy_state_dictionary.pt")
            prev_policy.load_state_dict(self.state_dict())
            return prev_policy

class Memory():
    def __init__(self):
        self.rewards = []
        self.eps_frames = []
        self.eps_frames_raw = []
        self.eps_mes = []
        self.eps_mes_raw = []
        self.actions = []
        self.actions_log_probs = []
        self.states_p = []
        self.terminals = []

    def add(self,frame,mes,raw_frame,raw_mes,a,a_log_prob,reward,s_prime,done):
        self.eps_frames.append(frame.detach().clone())
        self.eps_frames_raw.append(copy.deepcopy(raw_frame))
        self.eps_mes.append(mes)
        self.eps_mes_raw.append(copy.deepcopy(raw_mes))
        self.actions.append(a.detach().clone())
        self.actions_log_probs.append(a_log_prob.detach().clone())
        self.rewards.append(copy.deepcopy(reward))
        self.states_p.append(copy.deepcopy(s_prime))
        self.terminals.append(copy.deepcopy(done))

    def clear (self):
        #self.rewards = list(self.rewards.numpy())
        #self.actions_log_probs = list(self.actions_log_probs.numpy())

        self.rewards.clear()
        self.eps_frames.clear()
        self.eps_frames_raw.clear()
        self.eps_mes.clear()
        self.eps_mes_raw.clear()
        self.actions.clear()
        self.actions_log_probs.clear()
        self.states_p.clear()
        self.terminals.clear()

def train_model():
    env = gym.make("LunarLanderContinuous-v2")

    n_iters = 10000
    n_epochs = 50
    max_steps = 2000
    gamma = 0.99
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    moving_avg = 0

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_std = 0.5

    #init models
    policy = PPO_Agent(n_states, n_actions, action_std,lr, gamma, n_epochs,clip_val,device).to(device)
    prev_policy = PPO_Agent(n_states, n_actions, action_std,lr, gamma, n_epochs,clip_val,device).to(device)
    prev_policy.load_state_dict(policy.state_dict())
    memory = Memory()

    batch_ep_returns = []
    timestep_mod = 0
    total_timesteps = 0
    train_iters = 0
    update_timestep = 4000

    for iters in range(n_iters):
        s = env.reset()
        t = 0
        episode_return = 0
        done = False

        while not done:
            #s = prev_policy.format_state(s)
            a, a_log_prob = prev_policy.choose_action(s)
            s_prime, reward, done, info = env.step(a.detach().tolist()[0])

            memory.add(prev_policy.format_state(s), None ,None ,None,prev_policy.format_state(a),a_log_prob,reward,s_prime,done)

            s = copy.deepcopy(s_prime)
            t += 1
            total_timesteps +=1
            episode_return += reward

        # TODO change this hack to calculate when PPO training is triggered, look at PPO batch
        batch_ep_returns.append(episode_return)
        prev_timestep_mod = timestep_mod
        timestep_mod = total_timesteps // update_timestep

        if timestep_mod > prev_timestep_mod:
            prev_policy = policy.train(memory, prev_policy,iters)
            print ("on iter " + str(iters) + " with avg batch return " + str(np.array(batch_ep_returns).mean()))
            avg_batch_ep_returns = sum(batch_ep_returns)/len(batch_ep_returns)
            moving_avg = (avg_batch_ep_returns - moving_avg) * (2 / (train_iters + 2)) + avg_batch_ep_returns
            train_iters += 1
            batch_ep_returns.clear()
            memory.clear()


train_model()
