# pylint: skip-file
import sys, argparse
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
import datatable as dt

# instructions
# pip uninstall xgboost # rm -rf any residual site-packages/xgboost* in your environment
# git clone https://github.com/h2oai/xgboost
# git checkout h2oai_dt
# cd xgboost ; mkdir -p build ; cd build ; cmake .. -DUSE_CUDA=ON ; make -j ; cd ..
# cd python-package ; python setup.py install ; cd .. # installs as egg instead of like when doing wheel
# python tests/benchmark/testdt.py --algorithm=gpu_hist # use hist if you don't have a gpu


def run_benchmark(args, gpu_algorithm, cpu_algorithm):
    print("Generating dataset: {} rows * {} columns".format(args.rows, args.columns))
    print("{}/{} test/train split".format(args.test_size, 1.0 - args.test_size))
    tmp = time.time()
    X, y = make_classification(args.rows, n_features=args.columns, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=7)
    print ("Generate Time: %s seconds" % (str(time.time() - tmp)))

    param = {'objective': 'binary:logistic',
             'max_depth': 6,
             'silent': 0,
             'n_gpus': 1,
             'gpu_id': 0,
             'eval_metric': 'error',
             'debug_verbose': 0,
             }

    param['tree_method'] = gpu_algorithm

    do_dt = True
    do_ccont = True
    do_nondt = True

    tmp = time.time()
    if do_ccont:
        X_train_cc = X_train
        X_test_cc = X_test
        y_train_cc = y_train
        y_test_cc = y_test
    else:
        # convert to dt as test
        X_train_cc = np.asfortranarray(X_train)
        X_test_cc = np.asfortranarray(X_test)
        y_train_cc = np.asfortranarray(y_train)
        y_test_cc = np.asfortranarray(y_test)

        if not (X_train_cc.flags['F_CONTIGUOUS'] and X_test_cc.flags['F_CONTIGUOUS'] \
                        and y_train_cc.flags['F_CONTIGUOUS'] and y_test_cc.flags['F_CONTIGUOUS']):
            ValueError("Need data to be Fortran (i.e. column-major) contiguous")
    print("dt prepare1 Time: %s seconds" % (str(time.time() - tmp)))


    if do_nondt:
        print("np->DMatrix Start")
        # omp way
        tmp = time.time()
        # below takes about 2.826s if do_ccont=False
        # below takes about 0.248s if do_ccont=True
        dtrain = xgb.DMatrix(X_train_cc, y_train_cc, nthread=-1)
        print ("np->DMatrix1 Time: %s seconds" % (str(time.time() - tmp)))
        tmp = time.time()
        dtest = xgb.DMatrix(X_test_cc, y_test_cc, nthread=-1)
        print ("np->DMatrix2 Time: %s seconds" % (str(time.time() - tmp)))

        print("Training with '%s'" % param['tree_method'])
        tmp = time.time()
        xgb.train(param, dtrain, args.iterations, evals=[(dtest, "test")])
        print("Train Time: %s seconds" % (str(time.time() - tmp)))
    if do_dt:

        # convert to column-major contiguous in memory to mimic persistent column-major state
        # do_cccont = True leads to prepare2 time of about 1.4s for 1000000 rows * 50 columns
        # do_cccont = False leads to prepare2 time of about 0.000548 for 1000000 rows * 50 columns
        tmp = time.time()
        dtdata_X_train = dt.DataTable(X_train_cc)
        dtdata_X_test = dt.DataTable(X_test_cc)
        dtdata_y_train = dt.DataTable(y_train_cc)
        dtdata_y_test = dt.DataTable(y_test_cc)
        print ("dt prepare2 Time: %s seconds" % (str(time.time() - tmp)))

        #test = dtdata_X_train.tonumpy()
        #print(test)

        print ("dt->DMatrix Start")
        # omp way
        tmp = time.time()
        # below takes about 0.47s - 0.53s independent of do_ccont
        dtrain = xgb.DMatrix(dtdata_X_train, dtdata_y_train, nthread=-1)
        print ("dt->DMatrix1 Time: %s seconds" % (str(time.time() - tmp)))
        tmp = time.time()
        dtest = xgb.DMatrix(dtdata_X_test, dtdata_y_test, nthread=-1)
        print ("dt->DMatrix2 Time: %s seconds" % (str(time.time() - tmp)))

        print("Training with '%s'" % param['tree_method'])
        tmp = time.time()
        xgb.train(param, dtrain, args.iterations, evals=[(dtest, "test")])
        print ("Train Time: %s seconds" % (str(time.time() - tmp)))


parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=['all', 'gpu_exact', 'gpu_hist'], default='all')
parser.add_argument('--rows', type=int, default=1000000) # 1000000
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
