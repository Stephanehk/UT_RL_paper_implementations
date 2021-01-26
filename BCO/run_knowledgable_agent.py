#https://stable-baselines.readthedocs.io/en/master/modules/trpo.html
import gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make('CartPole-v1')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
D_demo = []
A_demo = []

state = env.reset()
for i in range(70000):
    action, _states = model.predict(state, deterministic=True)
    state_prime, reward, done, info = env.step(action)
    #env.render()

    D_demo.append([state, state_prime])
    A_demo.append(action)
    state = state_prime
    if done:
      state = env.reset()

env.close()
np.save("D_demo", D_demo)
np.save("A_demo", A_demo)
