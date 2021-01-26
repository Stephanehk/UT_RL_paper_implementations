import numpy as np
import cv2
import carla
import argparse
import logging
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import wandb
import math
import random

SHOW_PREVIEW = False;
#maybe scale these down
height = 480
width = 640
fov = 10
max_ep_length = 10
FPS = 60

# set_host = None
# set_world_port = None
# set_tm_port = None

class CarEnv:
    rgb_cam = None

    def __init__(self,host,world_port):
        self.client = carla.Client(host,world_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.car_agent_model = self.blueprint_lib.filter("model3")[0]

    def process_img(self,img):
        img = np.array(img.raw_data).reshape(height,width,4)
        rgb = img[:,:,:3]
        #norm = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.rgb_cam = rgb

    def reset(self):
        self.collisions = []
        self.actors = []
        #spawn car randomly
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.car_agent = self.world.spawn_actor(self.car_agent_model,self.spawn_point)
        self.actors.append(self.car_agent)
        #get camera
        self.rgb_cam = self.blueprint_lib.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x",f"{width}")
        self.rgb_cam.set_attribute("image_size_y",f"{height}")
        self.rgb_cam.set_attribute("fov",f"{fov}")
        sensor_pos = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, sensor_pos, attach_to=self.car_agent)
        self.actors.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        #workaround to get things started sooner
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)
        #get collision sensor
        col_sensor = self.blueprint_lib.find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_sensor,sensor_pos,attach_to=self.car_agent)
        self.actors.append(self.col_sensor)
        col_sensor.listen(lambda event: self.collision_data(event))

        while self.rgb_cam is None:
            print ("camera is not starting!")
            time.sleep(0.01)

        self.episode_start = time.time()
        #workaround to get things started sooner
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        return self.rgb_cam

    def step (self, action):
        self.vehicle.apply_control(carla.VehicleControl(throttle=action[0],steer=action[1]))
        velocity = self.vehicle.get_velocity()
        if (len(self.collisions) != 0):
            done = True
            reward = -100
        #velocity_kmh = int(3.6*np.sqrt(np.power(velocity.x,2) + np.power(velocity.y,2) + np.power(velocity.z,2)))
        # elif velocity_kmh < 40: #might stop car from going in circle
        #     reward = -1
        else:
            done = False
            reward = 1
        if self.episode_start + max_ep_length < time.time():
            done = True
        return self.rgb_cam, reward, done, None

    def cleanup():
        for actor in actors:
            actor.destroy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PPO_Agent(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(PPO_Agent, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Conv2d(3,6,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(6,12,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                nn.Linear(12*5*5, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Conv2d(3,6,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(6,12,5),
                nn.Tanh(),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                nn.Linear(12*5*5, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def choose_action(self,state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        mean = self.actor(state)
        cov_matrix = torch.diag(self.action_var).to(device)

        gauss_dist = MultivariateNormal(mean,cov_matrix)
        action = gauss_dist.sample()
        action_log_prob = gauss_dist.log_prob(action)
        return action, action_log_prob

    def get_training_params(self, state, action):
        state = torch.stack(state)
        action = torch.stack(action)

        mean = self.actor(state)
        action_expanded = self.action_var.expand_as(mean)
        cov_matrix = torch.diag_embed(action_expanded).to(device)

        gauss_dist = MultivariateNormal(mean,cov_matrix)
        action_log_prob = gauss_dist.log_prob(action)
        entropy = gauss_dist.entropy()
        state_value = torch.squeeze(self.critic(state))
        return action_log_prob, state_value, entropy

def format_(state):
    state = torch.FloatTensor(state).to(device)
    h,w,c = state.shape
    state = state.unsqueeze(0).view(1, c, h, w)
    return state

def train_PPO(host,world_port):
    #wandb.init(project='PPO_Carla_Attempt_1')

    env = CarEnv(host,world_port)
    n_iters = 1000
    n_epochs = 100
    max_steps = 2000
    gamma = 0.9
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    avg_r = 0

    config = wandb.config
    config.learning_rate = lr


    n_states = None

    #currently the action array will be [throttle, steer]
    n_actions = 2

    action_std = 0.5 #maybe try some other values for this
    #init models
    policy = PPO_Agent(n_states, n_actions, action_std).to(device)
    optimizer = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    prev_policy = PPO_Agent(n_states, n_actions, action_std).to(device)
    #TODO: idk if I should be setting each policies initial states to the same thing or not
    prev_policy.load_state_dict(policy.state_dict())


    #wandb.watch(prev_policy)

    for i in range (n_iters):
        s = env.reset()
        t = 0
        episode_reward = 0
        rewards = []
        states = []
        actions = []
        actions_log_probs = []
        states_p = []
        while t < max_steps:
            #env.render()
            a, a_log_prob = prev_policy.choose_action(s)
            s_prime, reward, done, info = env.step(a.detach())

            states.append(s)
            actions.append(format_(a))
            actions_log_probs.append(a_log_prob)
            rewards.append(reward)
            states_p.append(s_prime)

            s = s_prime
            t+=1
            episode_reward+=reward
            if done:
                break
        print ("Episode reward: " + str(episode_reward))
        avg_t+=t
        avg_r+=episode_reward

        #wandb.log({"episode_reward": np.absolute(episode_reward)})

        #format reward
        for i in range (len(rewards)):
            rewards[len(rewards)-1] = rewards[len(rewards)-1]*np.power(gamma,i)
        rewards = torch.tensor(rewards).to(device)
        rewards= (rewards-rewards.mean())/rewards.std()

        actions_log_probs = torch.FloatTensor(actions_log_probs).to(device)
        #train PPO
        for i in range(n_epochs):
            current_action_log_probs, state_values, entropies = policy.get_training_params(states,actions)

            policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())
            #policy_ratio = current_action_log_probs.detach()/actions_log_probs
            advantage = rewards - state_values.detach()

            update1 = policy_ratio*advantage
            update2 = torch.clamp(policy_ratio,1-clip_val, 1+clip_val) * advantage
            loss = float(-torch.min(update1,update2) + 0.5*mse(state_values,rewards) - 0.001*entropies)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            #wandb.log({"loss": loss})

        prev_policy.load_state_dict(policy.state_dict())

def main(n_vehicles,host,world_port,tm_port):
    #does nothing with n_vehicles and tm_port right now
    set_host = host
    set_world_port = world_port
    set_tm_port = tm_port

    train_PPO(host,world_port)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world_port', type=int, required=True)
    parser.add_argument('--tm_port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)

    main(**vars(parser.parse_args()))
