from __future__ import division
from pdb import set_trace as st
import torch.nn as nn
from dni_modules import DNIModule

class SynMLP(nn.Module):
  def __init__(self, args, in_features):
    super(SynMLP, self).__init__()
    self._linears = nn.ModuleList()
    if args.syn_batch_norm:
      self._batch_norms = nn.ModuleList()
    else:
      self._batch_norms = None
    shapes = zip((in_features,) + args.syn_units, args.syn_units + (in_features,))
    for index, shape in enumerate(shapes):
      linear = nn.Linear(*shape)
      self._linears.append(linear)
      if args.syn_batch_norm and index != len(shapes) - 1:
        batch_norm = nn.BatchNorm1d(shape[1])
        self._batch_norms.append(batch_norm)
    self._nonlinear = args.syn_nonlinear

  def initialize_network(self):
    for key, value in self.named_parameters():
      if 'linear' in key:
        value.data[:] = 0

  def forward(self, data):
    for index, linear in enumerate(self._linears):
      data = linear(data)
      if index != len(self._linears) - 1:
        if self._batch_norms:
          data = self._batch_norms[index](data)
        if self._nonlinear:
          data = self._nonlinear(data)
    return data

class DNILinear(DNIModule):
  def __init__(self, args, nonlinear, in_features, out_features):
    super(DNILinear, self).__init__(args.p_update)
    self._linear = nn.Linear(in_features, out_features)
    if args.mlp_batch_norm:
      self._batch_norm = nn.BatchNorm1d(out_features)
    else:
      self._batch_norm = None
    self._nonlinear = nonlinear
    self._synthesizer = SynMLP(args, in_features)
    self._synthesizer.initialize_network()
    syn_parameters = self._synthesizer.parameters()
    self._syn_optimizer = args.optimizer(syn_parameters, lr=1e-3)

  def _forward(self, data):
    data = self._linear(data)
    if self._batch_norm:
      data = self._batch_norm(data)
    if self._nonlinear:
      data = self._nonlinear(data)
    return data
  
  def _backward(self, data):
    gradient = self._synthesizer(data)
    return gradient

  def update_dni(self, cache):
    loss = self.dni_loss(cache)
    self._syn_optimizer.zero_grad()
    loss.backward()
    self._syn_optimizer.step()
    return loss

class DNI_MLP(nn.Module):
  def __init__(self, args):
    super(DNI_MLP, self).__init__()
    self._linears = nn.ModuleList()
    shapes = zip((28 * 28,) + args.mlp_units, args.mlp_units + (10,))
    for shape in shapes[:-1]:
      linear = DNILinear(args, args.mlp_nonlinear, *shape)
      self._linears.append(linear)
    classifier = DNILinear(args, None, *shapes[-1])
    self._linears.append(classifier)

    self._forward_caches = [None] * len(self._linears)
    self._backward_caches = [None] * len(self._linears)

  def forward(self, data):
    for index, linear in enumerate(self._linears):
      data, cache = linear(data)
      self._forward_caches[index] = cache
    return data

  def backward(self, gradient):
    for index, linear in enumerate(reversed(self._linears)):
      cache = self._forward_caches[len(self._linears) - index - 1]
      gradient, cache = linear.backward(gradient, cache)
      self._backward_caches[index] = cache

  def update_dni(self):
    loss_list = []
    for linear, cache in zip(self._linears, self._backward_caches):
      loss = linear.update_dni(cache)
      loss_list.append(loss)
    loss = sum(loss_list) / len(loss_list)
    return loss
