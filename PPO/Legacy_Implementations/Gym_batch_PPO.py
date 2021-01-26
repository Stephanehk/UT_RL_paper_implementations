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
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import kl
import gym
import copy

class PPO_Agent(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, gamma, n_epochs,clip_val,device):
        """
        Initializes PPO actor critic models
        """
        # action mean range -1 to 1
        super(PPO_Agent, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )

        #ppo training parameters
        self.batch_size = 128
        self.train_epoches = 4
        self.lr_a = 0.002
        self.lr_c = 0.005
        self.clip_norm = 0.5
        self.gamma = 0.99
        self.lam = 0.95

        self.optimizer_a = Adam(self.parameters(), lr=self.lr_a)
        self.optimizer_c = Adam(self.parameters(), lr=self.lr_c)

        self.mse = nn.MSELoss()
        #self.lr = lr
        self.n_epochs = n_epochs
        self.clip_val = clip_val
        self.device = device

        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)

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

    def format_state (self,s):
        """
        Input: raw state (nparray image and list of measurements)
        Output: frame and measurements (as tensors)
        """
        return torch.FloatTensor(s.reshape(1, -1))

    def estimate_advantage(self,memory):
        v_pred = memory.se_state_values
        rewards = memory.se_rewards
        #https://github.com/lilianweng/deep-reinforcement-learning-gym/blob/4fec4876ad28fe83309efd2cdf2a6f4281a5b23c/playground/policies/ppo.py#L173
        T = len(rewards)
        # Compute TD errors
        td_errors = [rewards[t] + self.gamma * v_pred[t + 1] - v_pred[t] for t in range(T - 1)]
        td_errors += [rewards[T - 1] + self.gamma * 0.0 - v_pred[T - 1]]  # handle the terminal state.

        assert len(memory.se_actions_log_probs) == len(v_pred) == len(td_errors) == T

        # Estimate advantage backwards.
        advs = []
        adv_so_far = 0.0
        for delta in td_errors[::-1]:
            adv_so_far = delta + self.gamma * self.lam * adv_so_far
            advs.append(adv_so_far)
        advs = advs[::-1]
        assert len(advs) == T
        return advs, advs+v_pred

    def train(self,memory,iters):
        """
        Input: memory object, previous policy, current iteration
        Output: updated previous policy (ie: current policy)
        """
        mean_entropies = []
        total_critic_loss = []
        total_actor_loss = []
        for n in range (self.train_epoches):
            eps_frames, eps_mes,actions,actions_log_probs,state_values,rewards,terminals,advantages,v_targets= memory.reservoir_sample(self.batch_size)

            actions_log_probs = torch.FloatTensor(actions_log_probs).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            state_values = torch.FloatTensor(state_values).to(self.device)
            v_targets = torch.FloatTensor(v_targets).to(self.device)

            #train PPO
            for i in range(self.n_epochs):
                current_action_log_probs, _, entropies = self.get_training_params(eps_frames, actions)
                policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())

                #update actor
                adv_l_update1 = policy_ratio*advantages
                adv_l_update2 = (torch.clamp(policy_ratio, 1-self.clip_val, 1+self.clip_val) * advantages).float()
                adv_l = torch.min(adv_l_update1, adv_l_update2)
                self.optimizer_a.zero_grad()
                adv_l.mean().backward()
                #clip gradient
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
                self.optimizer_a.step()

                #update critic
                #TODO: double check this is the same as loss_c = tf.reduce_mean(tf.square(self.v_target - self.critic))
                state_values.requires_grad = True
                v_targets.requires_grad = True


                loss_v = self.mse(state_values.float(), v_targets.float())
                self.optimizer_c.zero_grad()
                loss_v.backward()
                #clip gradient
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
                self.optimizer_c.step()

                #total loss (just for logging if necessary)
                total_loss = \
                    - adv_l \
                    + (0.5 * loss_v) \
                    - (0.01 * entropies)

                mean_entropies.append(entropies.detach().numpy().mean())
                total_critic_loss.append(loss_v.detach().numpy())
                total_actor_loss.append(adv_l.mean().detach().numpy())

            #
            # print("    on epoch " + str(n))
        # print (mean_entropies)
        # print (np.array(total_critic_loss).mean())
        # print (np.array(total_actor_loss).mean())

        if iters % 50 == 0:
            torch.save(self.state_dict(), "vanilla_policy_state_dictionary.pt")

        return np.array(mean_entropies).mean()

class Memory:
    def __init__(self):
        #data for entire rollout
        self.rewards = []
        self.eps_frames = []
        self.eps_frames_raw = []
        self.eps_mes = []
        self.eps_mes_raw = []
        self.actions = []
        self.actions_log_probs = []
        self.state_values = []
        self.states_p = []
        self.terminals = []
        self.advantages = []
        self.targets = []

        #single episdoe data
        self.se_rewards = []
        self.se_eps_frames = []
        self.se_eps_frames_raw = []
        self.se_eps_mes = []
        self.se_eps_mes_raw = []
        self.se_actions = []
        self.se_actions_log_probs = []
        self.se_state_values = []
        self.se_states_p = []
        self.se_terminals = []

    def add(self, frame, a, a_log_prob,state_values, reward, s_prime, done):
        self.eps_frames.append(frame)
        self.actions.append(a)
        self.actions_log_probs.append(a_log_prob.detach().clone())
        self.state_values.append(state_values.detach().clone())
        self.rewards.append(copy.deepcopy(reward))
        self.states_p.append(copy.deepcopy(s_prime))
        self.terminals.append(copy.deepcopy(done))

    def se_add(self, frame, a, a_log_prob,state_values, reward, s_prime, done):
        self.se_eps_frames.append(frame)
        self.se_actions.append(a)
        self.se_actions_log_probs.append(a_log_prob.detach().clone())
        self.se_state_values.append(state_values.detach().clone())
        self.se_rewards.append(copy.deepcopy(reward))
        self.se_states_p.append(copy.deepcopy(s_prime))
        self.se_terminals.append(copy.deepcopy(done))

    def add_advantages(self, advantages):
        self.advantages.extend(advantages)

    def add_targets (self,targets):
        self.targets.extend(targets)

    def clear(self):
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
        self.advantages.clear()
        self.targets.clear()
        self.state_values.clear()

    def se_clear(self):
        #self.se_rewards = list(self.se_rewards.numpy())
        #self.se_actions_log_probs = list(self.se_actions_log_probs.numpy())

        self.se_rewards.clear()
        self.se_eps_frames.clear()
        self.se_eps_frames_raw.clear()
        self.se_eps_mes.clear()
        self.se_eps_mes_raw.clear()
        self.se_actions.clear()
        self.se_actions_log_probs.clear()
        self.se_states_p.clear()
        self.se_terminals.clear()
        self.se_state_values.clear()

    def reservoir_sample(self, k):
        eps_frames_reservoir = []
        eps_mes_reservoir = []
        actions_reservoir = []
        actions_log_probs_reservoir = []
        rewards_reservoir = []
        terminals_reservoir = []
        advantages_reservoir = []
        state_values_reservoir = []
        targets_reservoir = []

        for i in range(len(self.eps_frames)):
            if len(eps_frames_reservoir) < k:
                eps_frames_reservoir.append(self.eps_frames[i])
                actions_reservoir.append(self.actions[i])
                actions_log_probs_reservoir.append(self.actions_log_probs[i])
                rewards_reservoir.append(self.rewards[i])
                terminals_reservoir.append(self.terminals[i])
                advantages_reservoir.append(self.advantages[i])
                state_values_reservoir.append(self.state_values[i])
                targets_reservoir.append(self.targets[i])
            else:
                j = int(random.uniform(0, i))
                if j < k:
                    eps_frames_reservoir[j] = self.eps_frames[i]
                    actions_reservoir[j] = self.actions[i]
                    actions_log_probs_reservoir[j] = self.actions_log_probs[i]
                    rewards_reservoir[j] = self.rewards[i]
                    terminals_reservoir[j] = self.terminals[i]
                    advantages_reservoir[j] = self.advantages[i]
                    state_values_reservoir[j] = self.state_values[i]
                    targets_reservoir[j] = self.targets[i]

        return eps_frames_reservoir, eps_mes_reservoir, actions_reservoir, actions_log_probs_reservoir,state_values_reservoir, rewards_reservoir, terminals_reservoir, advantages_reservoir,targets_reservoir

def train_PPO():
    #wandb.init(project='PPO_1')

    env = gym.make("LunarLanderContinuous-v2")
    n_iters = 2000
    n_epochs = 100
    max_steps = 1500
    gamma = 0.99
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    moving_avg = 0
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_std = 0.5
    n_rollout_workers = 5
    device = torch.device(f"cuda:{args.client_gpu}" if torch.cuda.is_available() else "cpu")

    #init models
    policy = PPO_Agent(n_states, n_actions, action_std, gamma, n_epochs, clip_val, device).to(device)
    # prev_policy = PPO_Agent(n_states, n_actions, action_std, gamma, n_epochs, clip_val, device).to(device)
    # prev_policy.load_state_dict(policy.state_dict())
    memory = Memory()

    batch_ep_returns = []
    timestep_mod = 0
    train_iters = 0
    total_timesteps = 0
    update_timestep = 4000
    prev_batch_return = 0
    avg_r = 0

    # config = wandb.config
    # config.learning_rate = lr

    rollout = 0
    for i in range (n_iters):
        s = env.reset()
        t = 0
        episode_return = 0
        while t < max_steps:
            env.render()
            a, a_log_prob = policy.choose_action(s)
            a_ = a.detach().numpy()[0]
            s_prime, reward, done, info = env.step(a_)

            s = policy.format_state(s)
            state_value = torch.squeeze(policy.critic(torch.squeeze(s)))
            memory.add(s, a, a_log_prob, state_value, reward, s_prime, done)
            memory.se_add(policy.format_state(s), a, a_log_prob, state_value, reward, s_prime, done)

            s = s_prime
            t+=1
            total_timesteps+=1
            episode_return+=reward
            if done:
                break
        #print ("Episode reward: " + str(episode_reward))
        avg_t+=t
        avg_r+=episode_return
        if i%20 == 0 and i > 0:
            print ("Average reward on iteration " + str(i) + ": " + str(avg_r/20))
            avg_r = 0

        #compute episode advtanages using the single episodes collected data
        advantages,v_targets = policy.estimate_advantage(memory)
        memory.add_advantages(advantages)
        memory.add_targets(v_targets)

        #clear single episodes collected data
        memory.se_clear()

        # TODO change this hack to calculate when PPO training is triggered, look at PPO batch
        batch_ep_returns.append(episode_return)
        prev_timestep_mod = timestep_mod
        timestep_mod = total_timesteps // update_timestep
        rollout += 1
        if rollout > n_rollout_workers:
            rollout = 0
            mean_entropy = policy.train(memory,i)
            memory.clear()

            avg_batch_ep_returns = sum(batch_ep_returns)/len(batch_ep_returns)
            moving_avg = (avg_batch_ep_returns - moving_avg) * (2 / (train_iters + 2)) + avg_batch_ep_returns
            train_iters += 1
            batch_ep_returns.clear()

            if avg_batch_ep_returns - prev_batch_return > 0.3:
                save_video = True
            else:
                save_video = False

            prev_batch_return = avg_batch_ep_returns
        # wandb.log({"episode_reward": episode_reward})
        # wandb.log({"n iters": i})
train_PPO()
