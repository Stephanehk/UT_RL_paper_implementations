import numpy as np
import gym
import torch
import random
from torch.autograd import Variable

def reservoir_sample(arr, k):
    reservoir = []
    for i in range (len(arr)):
        if len(reservoir) < k:
            reservoir.append(arr[i])
        else:
            j = int(random.uniform(0,i))
            if j < k:
                reservoir[j] = arr[i]
    return reservoir

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

def create_M_model(n_inputs, n_outputs):
    #init inverse dynamics model
    model = torch.nn.Sequential (
    torch.nn.Flatten(),
    torch.nn.Linear(n_inputs, 32),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(32, n_outputs),
    torch.nn.Softmax(dim=1)
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer

def create_policy_model (n_inputs, n_outputs):
    #init policy model
    model = torch.nn.Sequential (
    torch.nn.Linear(n_inputs, 32),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(32, n_outputs),
    torch.nn.Softmax(dim=0)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer

def BCO (alpha):

    def postrain_idm(idm_iters):
        #Learn inverse dynamics model
        transitions = []
        actions = []

        avg_t = 0
        for i in range (idm_iters):
            s = env.reset()
            t = 0
            while t < max_steps:
                #env.render()
                s_tens = torch.Tensor(s)
                #train M on post-deomnstration
                a = np.argmax(policy_model(s_tens).detach().numpy())
                s_prime, reward, done, info = env.step(a)
                transitions.append([list(s),list(s_prime)])
                actions.append(a)
                s = s_prime
                t+=1
                if done:
                    break
            avg_t+=t

        #format stuff
        transitions_samples = torch.Tensor(transitions)
        actions_samples = torch.Tensor(actions)
        actions_samples = actions_samples.long()

        #Update M
        print ("Avg iteration reward: " +str(avg_t/idm_iters))
        for epoch in range (1000):
            m_loss = m_loss_fn(m_model(transitions_samples), actions_samples)
            m_optimizer.zero_grad()
            m_loss.backward()
            m_optimizer.step()
        print ("IDM model loss: " + str(m_loss.item()))

    def pretrain_idm(idm_iters):
        #Learn inverse dynamics model
        transitions = []
        actions = []

        s = env.reset()
        for i in range (idm_iters):
            #env.render()
            s_tens = torch.Tensor(s)
            #train M on pre-deomnstration
            a = np.random.randint(n_actions)
            s_prime, reward, done, info = env.step(a)
            transitions.append([list(s),list(s_prime)])
            actions.append(a)
            s = s_prime
            if done:
                s = env.reset()
                break

        #format stuff
        transitions_samples = torch.Tensor(transitions)
        actions_samples = torch.Tensor(actions)
        actions_samples = actions_samples.long()

        #Update M
        for epoch in range (1000):
            m_loss = m_loss_fn(m_model(transitions_samples), actions_samples)
            m_optimizer.zero_grad()
            m_loss.backward()
            m_optimizer.step()
            print ("IDM model loss: " + str(m_loss.item()))

    env = gym.make("CartPole-v0")
    D_demo = np.load("D_demo.npy")
    num_episodes = 200
    max_steps = 200
    idm_iters = 10000

    n_actions = env.action_space.n
    n_states = len(env.reset())

    m_model, m_loss_fn, m_optimizer = create_M_model(2*n_states, n_actions)
    policy_model, policy_loss_fn, policy_optimizer = create_policy_model (n_states, n_actions)

    print ("initially training inverse dynamic model...")
    pretrain_idm(int(idm_iters/alpha))
    #while evaluate_idm(m_model, D_demo) < 0.85:
        #pretrain_idm(100000)
    evaluate_idm(m_model, D_demo)
    print ("training policy...")
    #start running episodes and actually improving
    for iter in range (num_episodes):

        #sample (s,s') pairs from learned agent
        D_demo_samples = reservoir_sample(D_demo,1000)

        #sample s from learned (s,s')
        S_demo_samples = torch.Tensor([S[0] for S in D_demo_samples])
        #use (s,s') pairs to estimate a
        A_demo_samples = [np.argmax(m_model(torch.Tensor([demo_transition])).detach().numpy()) for demo_transition in D_demo_samples]
        A_demo_samples = torch.Tensor(A_demo_samples)
        A_demo_samples = A_demo_samples.long()

        for epoch in range (500):
            policy_loss = policy_loss_fn(policy_model(S_demo_samples),A_demo_samples)
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
        print ("policy loss: " + str(policy_loss.item()))

        idm_iters = int(alpha*idm_iters)
        if idm_iters > 0:postrain_idm(idm_iters)
        else:postrain_idm(1)
        print ("\n")

        if iter % 100 == 0 or iter == num_episodes-1:
            torch.save(policy_model, "policy_model.pt")
            torch.save(m_model, "m_model.pt")

def test_BCO ():
    policy_model = torch.load("policy_model.pt")
    env = gym.make("CartPole-v0")
    test_iters = 100
    avg_t = 0
    max_steps = 200
    for i in range (test_iters):
        s = env.reset()
        t = 0
        while t < max_steps:
            env.render()
            s_tens = torch.Tensor(s)
            #train M on post-deomnstration
            a = np.argmax(policy_model(s_tens).detach().numpy())
            s_prime, reward, done, info = env.step(a)

            s = s_prime
            t+=1
            if done:
                break
        print ("Episode reward: " + str(t))
        avg_t+=t

BCO (0.1)
#BCO (0)
test_BCO ()
