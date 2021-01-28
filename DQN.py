'''
DQN with CartPole-v1
STATE:
    same as CartPole's observations https://github.com/openai/gym/wiki/CartPole-v0
ACTION:
    same as CartPole's actions https://github.com/openai/gym/wiki/CartPole-v0
REWARD：
    See code line 97
'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import gym 
from collections import namedtuple
import random

BATCH_SIZE = 32
LEARNING_RATE = 0.01  # learning rate
EPSILON = 0.9  #greedy policy
GAMMA = 0.9 # reward discount
MEMORY_CAPACITY = 200
env = gym.make('CartPole-v1').unwrapped
n_actions = env.action_space.n 
n_states = env.observation_space.shape[0]

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

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
        self.fc2 = nn.Linear(50,100)
        self.fc3 = nn.Linear(100, n_actions)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0),-1)

policy_net = Net()
target_net = Net()

def choose_action(s):
    if np.random.uniform() < EPSILON:
        s = torch.unsqueeze(s,0)
        actions_value = policy_net.forward(s)
        action = torch.max(actions_value, 1)[1].item()
    else:
        action = np.random.randint(0, n_actions)
    return torch.tensor([action])


def update():
    pass

if __name__ == "__main__":
    memory = ReplayMemory(MEMORY_CAPACITY)
    target_net.load_state_dict(policy_net.state_dict())
    num_episodes = 400
    learn_step = 1
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    for i in range(num_episodes):
        s = env.reset()
        s = torch.tensor(s, dtype=torch.float32)
        reward_sum = 0
        while True:
            env.render()
            a = choose_action(s) 
            s_next, r, done, _ = env.step(a.item()) 
            s_next = torch.tensor(s_next,dtype=torch.float32)

            # 定义新的奖赏函数，否则不收敛
            x, x_dot, theta, theta_dot = s_next
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1+r2
            r = torch.tensor([r],dtype=torch.float32)

            memory.push(s,a,r,s_next)
            s = s_next
            reward_sum += r.item()

            if len(memory) >= MEMORY_CAPACITY:

                transitons = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitons))

                
                s_batch = torch.cat(batch.state).view(BATCH_SIZE,-1)
                a_batch = torch.cat(batch.action).view(BATCH_SIZE,-1)
                r_batch = torch.cat(batch.reward).view(BATCH_SIZE,-1)
                s_next_batch = torch.cat(batch.next_state).view(BATCH_SIZE,-1)


                state_action_v = policy_net(s_batch).gather(1,a_batch) # 预测的Q值
                next_state_action_v = (1-done) * target_net(s_next_batch).max(1,keepdim=True)[0].detach()
                y = r_batch + GAMMA*next_state_action_v # 目标的Q值
                loss = criterion(y, state_action_v)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                print('episode{}--reward_sum{}'.format(i,round(reward_sum,2)))
                break



            





