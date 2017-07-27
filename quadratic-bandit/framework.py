import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn as nn

class Agent(nn.Module):
  def __init__(self, args):
    super(Agent, self).__init__()
    self._mean = nn.Parameter(th.rand(args.D).cuda())
    if args.pg:
      self._std = nn.Parameter(th.zeros(1).cuda())
    else:
      self._std = nn.Variable(th.ones(1).cuda() * args.std)
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
    a = th.normal(mean, std).detach()
    cache = (a, mean, std)
    return a, cache
  
  def _critic(self, r):
    return r

  def criterion(self, cache, r):
    a, mean, std = cache
    q = self._critic(r).expand_as(a)
    if self._pg:
      j = -(a - mean) ** 2 / (2 * std ** 2) * q
    j = th.mean(j)
    return j

class Environ(nn.Module):
  def __init__(self, args):
    super(Environ, self).__init__()
    eigs = np.random.uniform(args.least_eig, args.greatest_eig, (args.D))
    C = np.diag(eigs)
    C = th.from_numpy(C).float().cuda()
    self._C = Variable(C)
    a_star = np.ones((args.D,)) * args.a_star
    a_star = th.from_numpy(a_star).float().cuda()
    self._a_star = Variable(a_star)

  def reward(self, a):
    a = a - self._a_star.expand_as(a)
    quadratic = th.mm(th.mm(a, self._C), a.t())
    r = th.mean(quadratic)
    return r
