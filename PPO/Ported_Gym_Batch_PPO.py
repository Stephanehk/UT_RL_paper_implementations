import numpy as np
import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import wandb
import math
import random

gamma = 0.99
lam = 0.95
n_epochs = 80
n_batches = 1
clip_val = 0.2
lr = 0.0003
update_timestep = 4000
norm_adv = True
norm_return = False

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

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.mse = nn.MSELoss()

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

    def train_(self,memory,prev_policy):
        print ("updating")
        for b in range (n_batches):
            states,actions,actions_log_probs,returns,terminals,advantages= memory.reservoir_sample(len(memory.rewards))

            actions_log_probs = torch.FloatTensor(actions_log_probs)
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            #train PPO
            for i in range(n_epochs):
                current_action_log_probs, state_values, entropies = self.get_training_params(states,actions)

                policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())
                update1 = (policy_ratio*advantages).float()
                update2 = (torch.clamp(policy_ratio,1-clip_val, 1+clip_val) * advantages).float()
                loss = -torch.min(update1,update2) + 0.5*self.mse(state_values.float(),returns.float()) - 0.01*entropies

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                #wandb.log({"loss": loss})
        prev_policy.load_state_dict(self.state_dict())
        return prev_policy


class Memory():
    def __init__(self):
        self.rewards = []
        self.states = []
        self.actions = []
        self.actions_log_probs = []
        self.states_p = []
        self.terminals = []
        self.values = []

    def add (self,s,a,a_log_prob,r,s_prime,t,v):
        self.states.append(s)
        self.actions.append(a)
        self.actions_log_probs.append(a_log_prob)
        self.rewards.append(r)
        self.states_p.append(s_prime)
        self.terminals.append(t)
        self.values.append(v)

    def clear(self):
        #self.rewards = list(self.rewards.numpy())
        if torch.is_tensor(self.values):
            self.values = list(self.values.numpy())
        #self.actions_log_probs = list(self.actions_log_probs.numpy())
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.actions_log_probs.clear()
        self.states_p.clear()
        self.terminals.clear()
        self.values.clear()

    def reservoir_sample(self,k):
        #------------------------------TD Lambda GAE---------------------------------------------------------------------------
        #self.returns = [0 for i in range(len(self.rewards))]
        deltas = [0 for i in range(len(self.rewards))]
        self.advantages = [0 for i in range(len(self.rewards))]
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(len(self.rewards))):
            #self.returns[i] = self.rewards[i] + self.gamma * prev_return * self.terminals[i]
            #prev_return = self.returns[i]
        
            deltas[i] = self.rewards[i] + gamma * prev_value * self.terminals[i] - self.values[i]
            self.advantages[i] = deltas[i] + gamma * lam * prev_advantage * self.terminals[i]
            prev_value = self.values[i]
            prev_advantage = self.advantages[i]

        self.returns = np.array(self.advantages) - np.array(self.values)
        if norm_adv:
            self.advantages = (self.advantages-np.array(self.advantages).mean())/(np.array(self.advantages).std() + 1e-5)
        if norm_return:
            self.returns = (self.returns-np.array(self.returns).mean())/(np.array(self.returns).std() + 1e-5)
         #------------------------------TD Lambda GAE---------------------------------------------------------------------------
        returns_res = []
        states_res = []
        actions_res = []
        actions_log_probs_res = []
        terminals_res = []
        advantages_res = []
        for i in range (len(self.rewards)):
            if len(returns_res) < k:
                returns_res.append(self.returns[i])
                states_res.append(self.states[i])
                actions_res.append(self.actions[i])
                actions_log_probs_res.append(self.actions_log_probs[i])
                terminals_res.append(self.terminals[i])
                advantages_res.append(self.advantages[i])
            else:
                j = int(random.uniform(0,i))
                if j < k:
                    returns_res[j] = self.returns[i]
                    states_res[j] = self.states[i]
                    actions_res[j] = self.actions[i]
                    actions_log_probs_res[j] = self.actions_log_probs[i]
                    terminals_res[j] = self.terminals[i]
                    advantages_res[j] = self.advantages[i]
        return states_res,actions_res,actions_log_probs_res,returns_res,terminals_res,advantages_res


def format_(state):
    return torch.FloatTensor(state.reshape(1, -1))

def train_PPO():
    #wandb.init(project='PPO_1')

    env = gym.make("LunarLanderContinuous-v2")
    n_iters = 2000
    max_steps = 1500

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

    prev_policy = PPO_Agent(n_states, n_actions, action_std)
    prev_policy.load_state_dict(policy.state_dict())
    #wandb.watch(prev_policy)

    memory = Memory()

    for i in range (n_iters):
        s = env.reset()
        t = 0
        episode_reward = 0
        while t < max_steps:
            #env.render()
            a, a_log_prob = prev_policy.choose_action(s)
            a = a.detach().numpy()[0]
            #a = np.clip(a,1,-1)
            s_prime, reward, done, info = env.step(a)
            value = torch.squeeze(policy.critic(format_(s))).item()

            memory.add(format_(s),format_(a), a_log_prob,reward,s_prime,done,value)

            if total_timesteps % update_timestep == 0:
                print ("updating")
                prev_policy = policy.train_(memory,prev_policy)
                memory.clear()

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

train_PPO()
