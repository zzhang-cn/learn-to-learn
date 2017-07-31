from __future__ import division
import torch as th
from torch.autograd import Variable

class Statistics(object):
  def __init__(self):
    self._statistics = []

  def clear(self):
    self._statistics = []

  def _filter(self, s):
    if isinstance(s, Variable):
      s = s.data
    s = s[0]
    return s

  @property
  def statistics(self):
    raise NotImplementedError()

  def update(self, s):
    s = self._filter(s)
    self._statistics.append(s)

class Mean(Statistics):
  def __init__(self):
    super(Mean, self).__init__()

  @property
  def statistics(self):
    s = sum(self._statistics) / len(self._statistics)
    return s
