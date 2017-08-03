from pdb import set_trace as st
from argparse import ArgumentParser
import cPickle as pickle
import gzip
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from dni_modules import DNICriterion
from dni_networks import DNI_MLP

parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--mnist-path', type=str, default='mnist.pkl.gz')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--interval', type=int, default=100)
parser.add_argument('--n-epochs', type=int, default=10)
parser.add_argument('--n-units', type=str, default='1024-1024-1024')
parser.add_argument('--p-update', type=float, default=0.5)
args = parser.parse_args()
print args

if args.gpu > -1:
  cuda = True
  th.cuda.set_device(args.gpu)
else:
  cuda = False

partition = 'training', 'validation', 'test'
datasets = pickle.load(gzip.open(args.mnist_path, 'rb'))
datasets = dict(zip(partition, datasets))
data_loaders = {}
for key, value in datasets.items():
  value = map(th.from_numpy, value)
  dataset = TensorDataset(*value)
  shuffle = key == 'training'
  data_loaders[key] = DataLoader(dataset, args.batch_size, shuffle)

n_units = args.n_units.split('-')
n_units = map(int, n_units)
model = DNI_MLP(args.p_update, F.relu, 28 * 28, 10, *n_units)
if cuda:
  model.cuda()
criterion = DNICriterion(nn.CrossEntropyLoss())
optimizer = Adam(model.parameters(), lr=1e-3)
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

# for iteration, batch in enumerate(data_loaders['validation']):
