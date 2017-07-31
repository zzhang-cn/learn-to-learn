from pdb import set_trace as st

import math
import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class Actor(nn.Module):
  def __init__(self, args):
    super(Actor, self).__init__()
    cuda = lambda t: t if args.gpu < 0 else t.cuda()
    self._mean = nn.Parameter(cuda(th.rand(args.D)))
    if args.pg:
      std = th.ones(1) * math.log(args.std)
      self._std = nn.Parameter(cuda(std))
    else:
      std = th.ones(1) * args.std
      self._std = Variable(cuda(std))
    self._N = args.batch_size
    self._D = args.D
    self._pg = args.pg

  def forward(self):
    mean = self._mean.expand(self._N, self._D)
    if self._pg:
      std = th.exp(self._std)
    else:
      std = self._std
    std = std.expand_as(mean)
    action = th.normal(mean, std).detach()
    cache = (action, mean, std)
    return action, cache
  
  def criterion(self, cache, value):
    action, mean, std = cache
    value = value.expand_as(action)
    # detaching is unnecessary due to seperation of optimizers
    if self._pg:
      loss = -(action - mean) ** 2 / (2 * std ** 2) * value
    else:
      loss = value
    loss = th.mean(loss)
    return loss

class Critic(nn.Module):
  def __init__(self, args):
    super(Critic, self).__init__()
    self.criterion = nn.MSELoss()
    self._linear0 = nn.Linear(args.D, 16)
    self._linear1 = nn.Linear(16, 1)

  def forward(self, action):
    value = self._linear0(action)
    value = F.sigmoid(value)
    value = self._linear1(value)
    return value

class Agent(nn.Module):
  def __init__(self, args):
    super(Agent, self).__init__()
    self._actor = Actor(args)
    self._critic = Critic(args)
    if args.gpu > -1:
      self._actor.cuda()
      self._critic.cuda()
    self._actor_optimizer = Adam(self._actor.parameters(), lr=args.actor_lr)
    self._critic_optimizer = Adam(self._critic.parameters(), lr=args.critic_lr)
    self._pg = args.pg

  def forward(self):
    action, cache = self._actor()
    return action, cache

  def update(self, cache, penalty):
    action, mean, _ = cache
    if self._pg:
      value = self._critic(action)
    else:
      value = self._critic(mean)
    actor_loss = self._actor.criterion(cache, value)
    self._actor_optimizer.zero_grad()
    actor_loss.backward(retain_variables=True)
    self._actor_optimizer.step()

    if not self._pg:
      value = self._critic(action)
    critic_loss = self._critic.criterion(value, penalty)
    self._critic_optimizer.zero_grad()
    critic_loss.backward()
    self._critic_optimizer.step()

    return actor_loss, critic_loss

class Environ(nn.Module):
  def __init__(self, args):
    super(Environ, self).__init__()
    cuda = lambda t: t if args.gpu < 0 else t.cuda()
    eigs = np.random.uniform(args.least_eig, args.greatest_eig, (args.D))
    C = np.diag(eigs)
    C = cuda(th.from_numpy(C).float())
    self._C = Variable(C)
    a_star = th.ones(args.D,) * args.a_star
    self._a_star = Variable(cuda(a_star))

  def forward(self, a):
    # (a - a^*)^T \cdot C \cdot (a - a^*)
    a = a - self._a_star.expand_as(a)
    penalty = th.mm(a, self._C)
    penalty = th.unsqueeze(penalty, 1)
    a = th.unsqueeze(a, 2)
    penalty = th.bmm(penalty, a)
    penalty = th.squeeze(penalty).detach()
    return penalty
