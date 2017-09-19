# pylint: skip-file
import sys, argparse
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
import datatable as dt


def run_benchmark(args, gpu_algorithm, cpu_algorithm):
    print("Generating dataset: {} rows * {} columns".format(args.rows, args.columns))
    print("{}/{} test/train split".format(args.test_size, 1.0 - args.test_size))
    tmp = time.time()
    X, y = make_classification(args.rows, n_features=args.columns, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=7)
    print ("Generate Time: %s seconds" % (str(time.time() - tmp)))
    tmp = time.time()

    # convert to dt as test
    dtdata_X_train = dt.DataTable(X_train)
    dtdata_X_test = dt.DataTable(X_test)
    dtdata_y_train = dt.DataTable(y_train)
    dtdata_y_test = dt.DataTable(y_test)

    return
    print ("DMatrix Start")
    # omp way
    dtrain = xgb.DMatrix(X_train, y_train, nthread=-1)
    dtest = xgb.DMatrix(X_test, y_test, nthread=-1)
    print ("DMatrix Time: %s seconds" % (str(time.time() - tmp)))

    param = {'objective': 'binary:logistic',
             'max_depth': 6,
             'silent': 0,
             'n_gpus': 1,
             'gpu_id': 0,
             'eval_metric': 'error',
             'debug_verbose': 0,
             }

    param['tree_method'] = gpu_algorithm
    print("Training with '%s'" % param['tree_method'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations, evals=[(dtest, "test")])
    print ("Train Time: %s seconds" % (str(time.time() - tmp)))

    param['silent'] = 1
    param['tree_method'] = cpu_algorithm
    print("Training with '%s'" % param['tree_method'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations, evals=[(dtest, "test")])
    print ("Time: %s seconds" % (str(time.time() - tmp)))


parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=['all', 'gpu_exact', 'gpu_hist'], default='all')
parser.add_argument('--rows', type=int, default=10000) # 1000000
parser.add_argument('--columns', type=int, default=50)
parser.add_argument('--iterations', type=int, default=5) # 500
parser.add_argument('--test_size', type=float, default=0.25)
args = parser.parse_args()

if 'gpu_hist' in args.algorithm:
    run_benchmark(args, args.algorithm, 'hist')
elif 'gpu_exact' in args.algorithm:
    run_benchmark(args, args.algorithm, 'exact')
elif 'all' in args.algorithm:
    run_benchmark(args, 'gpu_exact', 'exact')
    run_benchmark(args, 'gpu_hist', 'hist')
