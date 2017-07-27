from argparse import ArgumentParser
import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
from framework import Agent, Environ

parser = ArgumentParser()
parser.add_argument('--a-star', type=float, default=4.0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--D', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--greatest-eig', type=float, default=1)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--least-eig', type=float, default=0.1)
parser.add_argument('--n_iterations', type=int, default=1000)
parser.add_argument('--pg', action='store_true', default=True)
parser.add_argument('--std', type=float, default=1)
args = parser.parse_args()
print args

th.cuda.set_device(args.gpu)

agent = Agent(args)
environ = Environ(args)

optimizer = Adam(agent.parameters(), lr=1e-3)
for i in range(args.n_iterations):
  action, cache = agent.forward()
  reward = environ.reward(action)
  j = agent.criterion(cache, reward)
  optimizer.zero_grad()
  j.backward()
  optimizer.step()

  if (i + 1) % args.interval == 0:
    print j.data[0]
