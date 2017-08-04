from __future__ import division
from pdb import set_trace as st
from argparse import Action, ArgumentParser
import cPickle as pickle
import gzip
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizers
from torch.utils.data import DataLoader, TensorDataset
from visdom import Visdom
from dni_modules import DNICriterion
from dni_networks import DNI_MLP
from utilities import count_errors
from visualizer import VisdomVisualizer

class to_nonlinear(Action):
  def __call__(self, parser, namespace, values, option_string):
    if hasattr(F, values):
      values = getattr(F, values)
    else:
      values = None
    setattr(namespace, self.dest, values)

class to_optimizer(Action):
  def __call__(self, parser, namespace, values, option_string):
    values = getattr(optimizers, values)
    setattr(namespace, self.dest, values)

def partition(delimiter, type):
  class action(Action):
    def __call__(self, parser, namespace, values, option_string):
      values = tuple(values.split(delimiter))
      values = map(type, values)
      setattr(namespace, self.dest, values)
  return action

parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--mnist-path', type=str, default='mnist.pkl.gz')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--mlp-batch-norm', action='store_true', default=True)
parser.add_argument('--mlp-nonlinear', type=str, default=F.relu, action=to_nonlinear)
parser.add_argument('--mlp-units', type=str, default=(256, 256, 256, 256), action=partition('-', int))
parser.add_argument('--n-epochs', type=int, default=10)
parser.add_argument('--optimizer', type=str, default=optimizers.Adam, action=to_optimizer)
parser.add_argument('--p-update', type=float, default=0.5)
parser.add_argument('--syn-nonlinear', type=str, default=F.relu, action=to_nonlinear)
parser.add_argument('--syn-batch-norm', action='store_true', default=True)
parser.add_argument('--syn-units', type=str, default=(1024, 1024), action=partition('-', int))
args = parser.parse_args()
print args

if args.gpu > -1:
  cuda = True
  th.cuda.set_device(args.gpu)
else:
  cuda = False

dataset_partition = 'training', 'validation', 'test'
datasets = pickle.load(gzip.open(args.mnist_path, 'rb'))
datasets = dict(zip(dataset_partition, datasets))
data_loaders = {}
for key, value in datasets.items():
  value = map(th.from_numpy, value)
  dataset = TensorDataset(*value)
  shuffle = key == 'training'
  data_loaders[key] = DataLoader(dataset, args.batch_size, shuffle)

model = DNI_MLP(args)
if cuda:
  model.cuda()
criterion = DNICriterion(nn.CrossEntropyLoss())
optimizer = args.optimizer(model.parameters(), lr=args.lr)
vis = Visdom()
dni_lvis = VisdomVisualizer(vis, {'title': 'dni loss'})
dni_llist = []
for epoch in range(args.n_epochs):
  for iteration, batch in enumerate(data_loaders['training']):
    data, labels = batch
    if cuda:
      data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    data = model(data)
    loss, cache = criterion(data, labels)
    optimizer.zero_grad()
    gradient = criterion.backward(cache)
    model.backward(gradient)
    optimizer.step()
    dni_loss = model.update_dni()
    dni_llist.append(dni_loss.data[0])

    if (iteration + 1) % args.interval == 0:
      print 'iteration %d loss %f' % (iteration + 1, loss.data[0])
      dni_lvis.extend(dni_llist, True)

  n_samples = 0
  n_errors = 0
  for batch in data_loaders['validation']:
    data, labels = batch
    if cuda:
      data, labels = data.cuda(), labels.cuda()
    data, labels = Variable(data), Variable(labels)
    data = model(data)
    n_samples += data.size()[0]
    n_errors += count_errors(data, labels)
  accuracy = 1 - n_errors / n_samples
  print 'epoch %d accuracy %f' % (epoch + 1, accuracy)
