import numpy as np
import gym
import torch


def create_idm_model(n_inputs, n_outputs):
    #init inverse dynamics model
    model = torch.nn.Sequential (
    torch.nn.Linear(n_inputs, 32),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(32, n_outputs),
    torch.nn.Softmax(dim=0)
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer

def RIDM():
    env = gym.make("Swimmer-v2")
    D_demo = np.load("D_demo.npy")

    num_episodes = 200
    episode_per_dem = 10
    #n_actions = env.action_space.n
    n_actions = 2
    n_states = len(env.reset())

    idm_model, idm_loss_fn, idm_optimizer = create_idm_model(2*n_states, n_actions)

    current_eps_i = 0
    current_epsiode = D_demo[current_eps_i]
    state = current_epsiode[0]
    n_eps_states = 0
    episode_reward = 0
    for eps in range(num_episodes):

        #TODO: THIS IS WHAT DOES NOT MAKE SENCE!!!
        if (n_eps_states+1 > len(current_epsiode)):
            n_eps_states = 0

        next_expert_state = current_epsiode[n_eps_states+1]

        #take action accoridng to IDM model
        state_pair = torch.flatten(torch.Tensor([state,next_expert_state]))
        action = idm_model(state_pair).detach().numpy()
        state_prime, reward, done, info = env.step(action)

        episode_reward+=reward

        state = state_prime
        n_eps_states+=1
        if done:
            env.reset()

            #TODO: Update IDM model parameters

            current_eps_i+=1
            current_epsiode = D_demo[int(current_eps_i/episode_per_dem)]
            state = current_epsiode[0]
            n_eps_states = 0
            episode_reward = 0

RIDM()
