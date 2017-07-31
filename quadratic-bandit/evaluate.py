from pdb import set_trace as st

from argparse import ArgumentParser
import numpy as np
import torch as th
from torch.autograd import Variable
import visdom
from framework import Agent, Environ
from utilities import Mean
from visualizer import Visualizer

parser = ArgumentParser()
parser.add_argument('--actor-lr', type=float, default=1e-3)
parser.add_argument('--a-star', type=float, default=4.0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--critic-lr', type=float, default=1e-3)
parser.add_argument('--D', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--greatest-eig', type=float, default=1)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--least-eig', type=float, default=0.1)
parser.add_argument('--n-iterations', type=int, default=1000)
parser.add_argument('--pg', action='store_true', default=False)
parser.add_argument('--std', type=float, default=1)
args = parser.parse_args()
print args

th.cuda.set_device(args.gpu)

agent = Agent(args)
environ = Environ(args)

metric = Mean()
vis = visdom.Visdom()
penalty_list = []
penalty_vis = Visualizer(vis, {'title': 'penalty'})
actor_llist = []
actor_lvis = Visualizer(vis, {'title': 'actor loss'})
critic_llist = []
critic_lvis = Visualizer(vis, {'title': 'critic loss'})

for i in range(args.n_iterations):
  action, cache = agent.forward()
  penalty = environ(action)
  actor_loss, critic_loss = agent.update(cache, penalty)

  penalty = th.mean(penalty)
  metric.update(penalty)
  penalty_list.append(penalty.data[0])
  actor_loss = th.mean(actor_loss)
  actor_llist.append(actor_loss.data[0])
  critic_loss = th.mean(critic_loss)
  critic_llist.append(critic_loss.data[0])

  if (i + 1) % args.interval == 0:
    print 'iteration #%d penalty %3f' % (i + 1, metric.statistics)
    penalty_vis.extend(penalty_list, True)
    actor_lvis.extend(actor_llist, True)
    critic_lvis.extend(critic_llist, True)

vis.close()
