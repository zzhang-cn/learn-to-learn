from argparse import ArgumentParser
import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F

parser = ArgumentParser()
parser.add_argument('--a-star', type=float, default=4.0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--D', type=int, default=10)
parser.add_argument('--greatest-eig', type=float, default=1)
parser.add_argument('--least-eig', type=float, default=0.1)
args = parser.parse_args()

eigs = np.random.uniform(args.least_eig, args.greatest_eig, (args.D))
C = np.diag(eigs)
C = Variable(C.cuda())
a_star = np.ones((args.D,)) * args.a_star
a_star = Variable(a_star.cuda())

def loss(a):
  normalized = a - a_star.expand_as(a)
  quadratic = th.linear(th.linear(normalized, C), normalized.t())
  error = th.mean(quadratic)
  return error
