from pdb import set_trace as st
import random
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class DNIModule(nn.Module):
  def __init__(self, p_update):
    super(DNIModule, self).__init__()
    self._p_update = p_update
    self._cache = {}

  def forward(self, data):
    in_variable = Variable(data.data, requires_grad=True)
    out_variable = self._forward(in_variable)
    cache = in_variable, out_variable
    return out_variable, cache

  def _forward(self, data):
    return NotImplementedError()

  def backward(self, gradient, cache):
    """ Authentic gradients might be derived from synthetic gradient. """
    in_variable, out_variable = cache
    out_variable.backward(gradient)
    # compute gradient w.r.t. parameters and input data
    authentic_gradient = Variable(in_variable.grad.data)
    data = Variable(in_variable.data)
    synthetic_gradient = self._backward(data)
    if random.random() < self._p_update:
      authentic = True
      gradient = authentic_gradient
    else:
      authentic = False
      gradient = synthetic_gradient
    gradient = gradient.data
    cache = authentic, authentic_gradient, synthetic_gradient
    return gradient, cache

  def _backward(self, data):
    return NotImplementedError()

  def dni_loss(self, cache):
    # TODO when to use authentic_gradient as target?
    _, authentic_gradient, synthetic_gradient = cache
    criterion = nn.MSELoss()
    loss = criterion(synthetic_gradient, authentic_gradient)
    return loss

class DNICriterion(object):
  def __init__(self, criterion):
    self._criterion = criterion

  def __call__(self, data, labels):
    in_variable = Variable(data.data, requires_grad=True)
    out_variable = self._criterion(in_variable, labels)
    cache = in_variable, out_variable
    return out_variable, cache

  def backward(self, cache):
    in_variable, out_variable = cache
    out_variable.backward()
    gradient = in_variable.grad.data
    return gradient
