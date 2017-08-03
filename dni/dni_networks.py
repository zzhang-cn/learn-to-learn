from __future__ import division
from pdb import set_trace as st
import torch.nn as nn
from dni_modules import DNILinear

class DNI_MLP(nn.Module):
  def __init__(self, p_update, nonlinear, in_features, out_features, *args):
    super(DNI_MLP, self).__init__()
    self._linears = nn.ModuleList()
    shapes = zip((in_features,) + args, args + (out_features,))
    for shape in shapes:
      linear = DNILinear(p_update, nonlinear, *shape)
      self._linears.append(linear)
    self._nonlinear = nonlinear

    self._forward_caches = [None] * len(self._linears)
    self._backward_caches = [None] * len(self._linears)

  def forward(self, data):
    for index, linear in enumerate(self._linears):
      data, cache = linear(data)
      self._forward_caches[index] = cache
      if index != len(self._linears) - 1:
        data = self._nonlinear(data)
    return data

  def backward(self, gradient):
    for index, linear in enumerate(reversed(self._linears)):
      cache = self._forward_caches[len(self._linears) - index - 1]
      gradient, cache = linear.backward(gradient, cache)
      self._backward_caches[index] = cache

  def dni_loss(self):
    loss_list = []
    for linear, cache in zip(self._linears, self._backward_caches):
      loss = linear.loss(cache)
      loss_list.append(loss)
    loss = sum(loss_list) / len(loss_list)
    return loss
