#pylint: skip-file
import sys, argparse
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
import ast

rng = np.random.RandomState(1994)


def run_benchmark(args):

    try:
        dtest = xgb.DMatrix('dtest.dm')
        dtrain = xgb.DMatrix('dtrain.dm')

        if not (dtest.num_col() == args.columns \
                and dtrain.num_col() == args.columns):
            raise ValueError("Wrong cols")
        if not (dtest.num_row() == args.rows * args.test_size \
                and dtrain.num_row() == args.rows * (1-args.test_size)):
            raise ValueError("Wrong rows")
    except:

        print("Generating dataset: {} rows * {} columns".format(args.rows, args.columns))
        print("{}/{} test/train split".format(args.test_size, 1.0 - args.test_size))
        tmp = time.time()
        X, y = make_classification(args.rows, n_features=args.columns, n_redundant=0, n_informative=args.columns, n_repeated=0, random_state=7)
        if args.sparsity < 1.0:
           X = np.array([[np.nan if rng.uniform(0, 1) < args.sparsity else x for x in x_row] for x_row in X])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=7)
        print ("Generate Time: %s seconds" % (str(time.time() - tmp)))
        tmp = time.time()
        print ("DMatrix Start")
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test, nthread=-1)
        print ("DMatrix Time: %s seconds" % (str(time.time() - tmp)))

        dtest.save_binary('dtest.dm')
        dtrain.save_binary('dtrain.dm')

    param = {'objective': 'binary:logistic','booster':'gblinear'}
    if args.params is not '':
        param.update(ast.literal_eval(args.params))

    param['updater'] = args.updater
    print("Training with '%s'" % param['updater'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations, evals=[(dtrain,"train")], early_stopping_rounds = args.columns)
    print ("Train Time: %s seconds" % (str(time.time() - tmp)))

parser = argparse.ArgumentParser()
parser.add_argument('--updater', default='coord_descent')
parser.add_argument('--sparsity', type=float, default=0.0)
parser.add_argument('--lambda', type=float, default=1.0)
parser.add_argument('--tol', type=float, default=1e-5)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--rows', type=int, default=1000000)
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--columns', type=int, default=50)
parser.add_argument('--test_size', type=float, default=0.25)
parser.add_argument('--standardise', type=bool, default=False)
parser.add_argument('--params', default='', help='Provide additional parameters as a Python dict string, e.g. --params \"{\'max_depth\':2}\"')
args = parser.parse_args()

run_benchmark(args)
