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
             'tree_method': 'exact',
             'max_depth': 6,
             'silent': 1,
             'eval_metric': 'auc'}

    param['updater'] = gpu_algorithm
    print("Training with '%s'" % param['updater'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations)
    print ("Time: %s seconds" % (str(time.time() - tmp)))

    param['updater'] = cpu_algorithm
    print("Training with '%s'" % param['updater'])
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations)
    print ("Time: %s seconds" % (str(time.time() - tmp)))



parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=['all', 'grow_gpu', 'grow_gpu_hist'], required=True)
parser.add_argument('--rows',type=int,default=1000000)
parser.add_argument('--columns',type=int,default=50)
parser.add_argument('--iterations',type=int,default=500)
args = parser.parse_args()

if 'grow_gpu_hist' in args.algorithm:
    run_benchmark(args, args.algorithm, 'grow_fast_histmaker')
if 'grow_gpu ' in args.algorithm:
    run_benchmark(args, args.algorithm, 'grow_colmaker')
if 'all' in args.algorithm:
    run_benchmark(args, 'grow_gpu', 'grow_colmaker')
    run_benchmark(args, 'grow_gpu_hist', 'grow_fast_histmaker')

