import gym
import numpy as np
import torch

def create_M_model(n_inputs, n_outputs):
    #init inverse dynamics model
    model = torch.nn.Sequential (
    torch.nn.Flatten(),
    torch.nn.Linear(n_inputs, 16),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(16,32),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(32, n_outputs),
    torch.nn.Softmax(dim=1)
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer

def evaluate_idm(m_model, D_demo):
    A_demo = np.load("A_demo.npy")
    #test idm model accuracy
    X = torch.Tensor(D_demo)
    y_preds = m_model(X)

    n_correct = 0
    for i in range (len(y_preds)):
        if np.argmax(y_preds[i].detach().numpy()) == A_demo[i]:
            n_correct+=1
    print ("Accuracy: " + str(n_correct/len(D_demo)))
    return n_correct/len(D_demo)

def test_idm():
    env = gym.make("CartPole-v0")
    D_demo = np.load("D_demo.npy")
    A_demo = np.load("A_demo.npy")
    num_episodes = 1
    max_steps = 200

    pretrain_iters = 100000
    n_actions = env.action_space.n
    n_states = len(env.reset())

    #start running episodes and actually improving
    for iter in range (num_episodes):
        m_model, m_loss_fn, m_optimizer = create_M_model(2*n_states, n_actions)

        # transitions = D_demo
        # actions = A_demo
        #Learn inverse dynamics model
        transitions = []
        actions = []

        s = env.reset()
        for i in range (pretrain_iters):
            #env.render()
            a = np.random.randint(n_actions)

            s_prime, reward, done, info = env.step(a)
            transitions.append([list(s),list(s_prime)])
            actions.append(a)
            s = s_prime
            if done:
                s = env.reset()
            #print ("on iteration " + str(i))
        #format stuff
        transitions_samples = torch.Tensor(transitions)
        actions_samples = torch.Tensor(actions)
        actions_samples = actions_samples.long()

        #Update M
        # for j in range (100):
        for epoch in range (10000):
            m_loss = m_loss_fn(m_model(transitions_samples), actions_samples)
            m_optimizer.zero_grad()
            m_loss.backward()
            m_optimizer.step()
            evaluate_idm(m_model, D_demo)
            print ("Loss: " + str(m_loss.item()))
        print ("\n")

test_idm()
