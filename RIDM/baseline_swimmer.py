#
#https://stable-baselines.readthedocs.io/en/master/modules/trpo.html
import gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make('Swimmer-v2')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
D_demo = []
A_demo = []
current_epsiode = []

state = env.reset()

for i in range(70000):
    current_epsiode.append(state)
    action, _states = model.predict(state, deterministic=True)
    #print (action)
    state_prime, reward, done, info = env.step(action)
    #env.render())
    A_demo.append(action)
    state = state_prime
    if done:
      current_epsiode.append(state)
      D_demo.append(current_epsiode)
      current_epsiode = []

      state = env.reset()
      #print ("finished game")
    if i%10000 == 0:
        print ("on iteration " + str(i))

env.close()
np.save("D_demo", D_demo)
np.save("A_demo", A_demo)
