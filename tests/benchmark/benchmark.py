# pylint: skip-file
import sys, argparse
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
import ast

rng = np.random.RandomState(1994)


def run_benchmark(args):
    print("Generating dataset: {} rows * {} columns".format(args.rows, args.columns))
    print("{}/{} test/train split".format(args.test_size, 1.0 - args.test_size))
    tmp = time.time()
    X, y = make_classification(args.rows, n_features=args.columns, random_state=7)
    if args.sparsity < 1.0:
       X = np.array([[np.nan if rng.uniform(0, 1) < args.sparsity else x for x in x_row] for x_row in X])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=7)
    print ("Generate Time: %s seconds" % (str(time.time() - tmp)))
    tmp = time.time()
    print ("DMatrix Start")
    dtrain = xgb.DMatrix(X_train, y_train, nthread=-1)
    dtest = xgb.DMatrix(X_test, y_test, nthread=-1)
    print ("DMatrix Time: %s seconds" % (str(time.time() - tmp)))

    param = {'objective': 'binary:logistic'}
    if args.params is not '':
        param.update(ast.literal_eval(args.params))

    param['tree_method'] = args.tree_method
    print("Training with '%s'" % param['tree_method'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations, evals=[(dtest, "test")])
    print ("Train Time: %s seconds" % (str(time.time() - tmp)))

parser = argparse.ArgumentParser()
parser.add_argument('--tree_method', default='gpu_hist')
parser.add_argument('--sparsity', type=float, default=0.0)
parser.add_argument('--rows', type=int, default=1000000)
parser.add_argument('--columns', type=int, default=50)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--test_size', type=float, default=0.25)
parser.add_argument('--params', default='', help='Provide additional parameters as a Python dict string, e.g. --params \"{\'max_depth\':2}\"')
args = parser.parse_args()

run_benchmark(args)
