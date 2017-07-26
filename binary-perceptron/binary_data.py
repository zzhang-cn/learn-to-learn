from argparse import ArgumentParser
import numpy as np
import joblib

parser = ArgumentParser()
parser.add_argument('--D', type=int, default=3)
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--range', type=float, default=1.0)
args = parser.parse_args()

shape = (args.N, args.D)
data = np.random.uniform(-args.range, args.range, shape)
criterion = np.sum(data, 1) > 0
labels = np.zeros((args.N,))
labels[criterion] = 1

path = 'binary-data-D-%d-N-%d-range-%f' % (args.D, args.N, args.range)
joblib.dump((data, labels), path)
