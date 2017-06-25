# pylint: skip-file
import sys, argparse
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
import time

n = 1000000
num_rounds = 500

def run_benchmark(args, gpu_algorithm, cpu_algorithm):
    print("Generating dataset: {} rows * {} columns".format(args.rows,args.columns))
    X, y = make_classification(args.rows, n_features=args.columns, random_state=7)
    dtrain = xgb.DMatrix(X, y)

    param = {'objective': 'binary:logistic',
             'max_depth': 6,
             'silent': 1,
             'eval_metric': 'auc'}

    param['tree_method'] = gpu_algorithm
    print("Training with '%s'" % param['tree_method'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations)
    print ("Time: %s seconds" % (str(time.time() - tmp)))

    param['tree_method'] = cpu_algorithm
    print("Training with '%s'" % param['tree_method'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations)
    print ("Time: %s seconds" % (str(time.time() - tmp)))



parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=['all', 'gpu_exact', 'gpu_hist'], default='all')
parser.add_argument('--rows',type=int,default=1000000)
parser.add_argument('--columns',type=int,default=50)
parser.add_argument('--iterations',type=int,default=500)
args = parser.parse_args()

if 'gpu_hist' in args.algorithm:
    run_benchmark(args, args.algorithm, 'hist')
if 'gpu_exact' in args.algorithm:
    run_benchmark(args, args.algorithm, 'exact')
if 'all' in args.algorithm:
    run_benchmark(args, 'gpu_exact', 'exact')
    run_benchmark(args, 'gpu_hist', 'hist')

