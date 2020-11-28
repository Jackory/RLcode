import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import gym 
from collections import namedtuple
import random

bach_size = 32
alpha = 0.9  # learning rate
epsilon = 0.9  #greedy policy
gamma = 0.9 # reward discount
memory_capacity = 200
env = gym.make('CartPole-v0').unwrapped
n_actions = env.action_space.n 
n_states = env.observation_space.shape[0]

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.fc1 = nn.Linear(n_states, 50)
        self.fc2 = nn.Linear(50, n_actions)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

policy_net = Net()
target_net = Net()

def choose_action(s):
    s = torch.tensor(s, dtype=torch.float32)
    if np.random.uniform() < epsilon:
        action = torch.argmax(policy_net(s)).data.numpy()
        return action
    else:
        return np.random.randint(0, n_actions)


def update():
    pass 

if __name__ == "__main__":
    memory = ReplayMemory(memory_capacity)

    target_net.load_state_dict(policy_net.state_dict())
    num_episodes = 50

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=alpha)
    for i in range(num_episodes):
        s = env.reset()
        reward_sum = 0
        while True:
            a = choose_action(s) 
            s_next, r, done, _ = env.step(a) 
            memory.push(s,a,r,s_next)
            s = s_next
            reward_sum += r
            if len(memory) >= memory_capacity:
                transitons = memory.sample(bach_size)
                batch = Transition(*zip(*transitons))
                s_batch = batch.state
                a_batch = batch.action
                r_batch = batch.reward
                s_next_batch = batch.next_state

                state_action_v = policy_net(s_batch).gather(1,a_batch) # TODO:
                next_state_action_v = target(s_next.batch).max(1)[0].detach()
                y = r_batch + gamma*next_state_action_v
                loss = nn.MSELoss(y,state_action_v)
                optimizer.zero_grad()
                loss.backgrad()
                optimizer.step()
        if done:
            print('episode{}--reward_sum{}').format(i,round(reward_sum,2))
            break



            





