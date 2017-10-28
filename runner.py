import numpy as np
import pandas as pd
import os
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import operator
import code
from functools import reduce
import argparse
import logging

from random_forest import WaveletsForestRegressor

def load_csv(file_path):
  return pd.read_csv(file_path, delimiter=',', header=None).values

def load_npz(file_path, name):
  return np.load(file_path)[name]

class LoadFromFile (argparse.Action):
  def __call__ (self, parser, namespace, values, option_string = None):
    with open(values) as f:
      parsed = parser.parse_args(f.read().split(), namespace)
      return parsed

def main():
  parser = argparse.ArgumentParser(description='WaveletsForestRegressor runner. Use "python -m pydoc random_forest" or see "random_forest.html" for more details.')
  default_config_path = 'config.txt'
  config_action = parser.add_argument('--config', default=default_config_path, action=LoadFromFile)
  parser.add_argument(
      '--log',
      default='INFO',
      help='Logging level. Default is INFO.')

  parser.add_argument(
      '--regressor',
      default='rf',
      help='Regressor type. Either "rf" or "decision_tree_with_bagging". Default is "rf".')

  parser.add_argument(
      '--trees',
      default=5,
      type=int,
      help='Number of trees in the forest. Default is 5.')

  parser.add_argument(
      '--features',
      default='auto',
      help='Features to consider in each split. Same options as sklearn\'s DecisionTreeRegressor.')

  parser.add_argument(
      '--depth',
      default=9,
      type=int,
      help='Maximum depth of each tree. Default is 9. Use 0 for unlimited depth.')

  parser.add_argument(
      '--seed',
      default=2000,
      type=int,
      help='Seed for random operations. Default is 2000.')

  parser.add_argument(
      '--criterion',
      default='mse',
      help='Splitting criterion. Same options as sklearn\'s DecisionTreeRegressor. Default is "mse".')

  parser.add_argument(
      '--bagging',
      default=0.8,
      type=float,
      help='Bagging. Only available when using the "decision_tree_with_bagging" regressor. Default is 0.8.')

  parser.add_argument(
      '--data',
      default='trainingData.csv',
      help='Training data path. Default is "trainingData.csv".')

  parser.add_argument(
      '--labels',
      default='trainingLabel.csv',
      help='Training labels path. Default is "trainingLabel.csv".')

  parser.add_argument(
      '--results',
      default='results',
      help='Results save path.')


  parser.add_argument(
      '--shell',
      default=False,
      help='Drop into python shell after calculating smoothness. Default is False.',
      action='store_true')


  flags, _ = parser.parse_known_args()
  if os.path.exists(flags.config):
      config_flags = config_action(parser, argparse.Namespace(), flags.config)
      aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
      for arg in vars(flags): aux_parser.add_argument('--'+arg)
      flags, _ = aux_parser.parse_known_args(namespace=config_flags)

  np.random.seed(flags.seed)

  logging.basicConfig(level=getattr(logging, flags.log))

  logging.info('Creating regressor with (regressor=%s, trees=%s, features=%s, depth=%s, seed=%s, criterion=%s, bagging=%s)' % (flags.regressor, flags.trees, flags.features, flags.depth, flags.seed, flags.criterion, flags.bagging) )

  if int(flags.depth) <= 0:
    flags.depth = None
  else:
    flags.depth = int(flags.depth)

  if flags.seed is not None:
    flags.seed = int(flags.seed)

  if flags.trees is not None:
    flags.trees = int(flags.trees)


  try:
    flags.features = int(flags.features)
  except ValueError:
    pass
  try:
    flags.features = float(flags.features)
  except ValueError:
    pass


  regressor = WaveletsForestRegressor(regressor=flags.regressor, trees=flags.trees, features=flags.features, seed=flags.seed, depth=flags.depth)

  logging.info('Loading data=%s and labels=%s' % (flags.data, flags.labels))
  X = None
  y = None
  if flags.data.endswith('csv'):
    X = load_csv(flags.data)
  if flags.labels.endswith('csv'):
    y = load_csv(flags.labels)

  if flags.data.endswith('npz'):
    X = load_npz(flags.data, 'data')
  if flags.labels.endswith('npz'):
    y = load_npz(flags.labels, 'labels')



  rf = regressor.fit(X, y)
  alpha, n_wavelets, errors = rf.evaluate_smoothness()

  results_path = flags.results
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  with open(results_path + '/alpha.txt', 'w') as f:
    f.write('%s' % alpha)
  np.savetxt(results_path + '/NwaveletsInWaveletByWaveletTraining.txt', n_wavelets, fmt='%s')
  np.savetxt(results_path + '/errorByWaveletsTraining.txt', errors, fmt='%s')

  if flags.shell:
    code.interact(local=dict(globals(), **locals()))

if '__main__' == __name__:
    main()
