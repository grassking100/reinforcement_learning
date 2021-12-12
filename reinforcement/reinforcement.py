#!pip install torch
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli
from matplotlib import pyplot as plt

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(4, 8)
        self.linear_2 = nn.Linear(8, 16)
        self.linear_3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.sigmoid(self.linear_3(x))
        return x
    
def calc_weighted_rewards(records):
    gamma = 0.99
    reward_sums = []
    for index in range(len(records)):
        reward_sum = 0
        for index_1,record in enumerate(records[index:]):
            reward_sum += record[2]*gamma**(index_1)
        reward_sums += [reward_sum]
    return (np.array(reward_sums) - np.mean(reward_sums))/ np.std(reward_sums)

def calc_loss(act,weighted_reward,prob):
    m = Bernoulli(prob)
    loss = -m.log_prob(torch.FloatTensor([act])) * weighted_reward
    return loss

def update_model(actor,optim,records):
    actor.train(True)
    optim.zero_grad()
    loss = 0
    weighted_rewards = calc_weighted_rewards(records)
    for weighted_reward,record in zip(weighted_rewards,records):
        act,obs = record[:2]
        prob = actor(torch.FloatTensor(obs))
        loss += calc_loss(act,weighted_reward,prob)
    loss.backward()
    optim.step()

def choice_act_by_actor(env,actor):
    with torch.no_grad():
        probs = actor(torch.FloatTensor(env.state))
        act = int(Bernoulli(probs).sample().numpy()[0])
    return act

def simulation_by_actor(env,actor):
    act = choice_act_by_actor(env,actor)
    obs,reward,done = env.step(act)[:3]
    return act,obs,reward,done

def train_step(env,actor,optim):
    records = full_simulation_by_actor(env,actor)
    update_model(actor,optim,records)
    return len(records)

def full_simulation_by_actor(env,actor,render=False):
    records = []
    preb_obs = env.reset()
    for j in range(1000):
        if render:
            env.render()
        act,obs,reward,done = simulation_by_actor(env,actor)
        if done:
            break
        records += [[act,preb_obs,reward]]
        preb_obs = obs
    return records

def train(actor):
    optim = torch.optim.RMSprop(actor.parameters(),lr=0.01)
    steps = []
    env = gym.make('CartPole-v0')
    for _ in range(500):
        steps += [train_step(env,actor,optim)]
    return steps

def test(actor):
    env = gym.make('CartPole-v0')
    actor.train(False)
    env.reset()
    steps = []
    for _ in range(100):
        records = full_simulation_by_actor(env,actor)
        steps += [len(records)]
    return steps
