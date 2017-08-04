import torch as th

def count_errors(p, labels):
  _, p = th.max(p, 1)
  p = th.squeeze(p)
  indicators = (p != labels).float()
  n = th.sum(indicators)
  n = n.data[0]
  n = int(n)
  return n
