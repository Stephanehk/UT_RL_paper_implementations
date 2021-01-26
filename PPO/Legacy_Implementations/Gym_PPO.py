import numpy as np
import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import wandb
import math


class PPO_Agent(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
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
        self.action_var = torch.full((action_dim,), action_std*action_std)

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

def format_(state):
    return torch.FloatTensor(state.reshape(1, -1))

def train_PPO():
    #wandb.init(project='PPO_1')

    env = gym.make("LunarLanderContinuous-v2")
    n_iters = 2000
    n_epochs = 100
    max_steps = 1500
    update_timestep = 4000
    gamma = 0.99
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    avg_r = 0
    total_timesteps = 1

    # config = wandb.config
    # config.learning_rate = lr

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    action_std = 0.5 #maybe try some other values for this
    #init models
    policy = PPO_Agent(n_states, n_actions, action_std)
    optimizer = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    prev_policy = PPO_Agent(n_states, n_actions, action_std)
    prev_policy.load_state_dict(policy.state_dict())
    #wandb.watch(prev_policy)

    rewards = []
    states = []
    actions = []
    actions_log_probs = []
    states_p = []
    terminals = []

    for i in range (n_iters):
        s = env.reset()
        t = 0
        episode_reward = 0
        while t < max_steps:
            env.render()
            a, a_log_prob = prev_policy.choose_action(s)
            a = a.detach().numpy()[0]
            s_prime, reward, done, info = env.step(a)

            states.append(format_(s))
            actions.append(format_(a))
            actions_log_probs.append(a_log_prob)
            rewards.append(reward)
            states_p.append(format_(s_prime))
            terminals.append(done)

            if total_timesteps % update_timestep == 0:
                print ("updating")
                #format reward
                discounted_reward = 0
                for i in range (len(rewards)):
                    if terminals[len(rewards)-1-i]:
                        discounted_reward = 0
                    rewards[len(rewards)-1-i] = rewards[len(rewards)-1-i] + (gamma*discounted_reward)
                    discounted_reward = rewards[len(rewards)-1-i]

                #print (rewards)
                rewards = torch.tensor(rewards)
                rewards= (rewards-rewards.mean())/(rewards.std() + + 1e-5)

                actions_log_probs = torch.FloatTensor(actions_log_probs)
                #train PPO
                for i in range(n_epochs):
                    current_action_log_probs, state_values, entropies = policy.get_training_params(states,actions)

                    policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())
                    #policy_ratio = current_action_log_probs.detach()/actions_log_probs
                    advantage = rewards - state_values.detach()

                    update1 = (policy_ratio*advantage).float()
                    update2 = (torch.clamp(policy_ratio,1-clip_val, 1+clip_val) * advantage).float()
                    loss = -torch.min(update1,update2) + 0.5*mse(state_values.float(),rewards.float()) - 0.01*entropies

                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    #wandb.log({"loss": loss})
                prev_policy.load_state_dict(policy.state_dict())

                rewards = list(rewards.numpy())
                actions_log_probs = list(actions_log_probs.numpy())
                del rewards[:]
                del states[:]
                del actions[:]
                del actions_log_probs[:]
                del states_p[:]
                del terminals[:]

            s = s_prime
            t+=1
            total_timesteps+=1
            episode_reward+=reward
            if done:
                break
        #print ("Episode reward: " + str(episode_reward))
        avg_t+=t
        avg_r+=episode_reward
        if i%20 == 0 and i > 0:
            print ("Average reward on iteration " + str(i) + ": " + str(avg_r/20))
            avg_r = 0
        # wandb.log({"episode_reward": episode_reward})
        # wandb.log({"n iters": i})



train_PPO()
