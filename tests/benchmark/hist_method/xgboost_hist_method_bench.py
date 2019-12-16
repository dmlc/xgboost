#*******************************************************************************
# Copyright 2017-2019 by Contributors
# \file xgboost_hist_method_bench.py
# \brief a benchmark for 'hist' tree_method on both CPU/GPU arhitectures
# \author Egor Smirnov
#*******************************************************************************

import argparse
import xgboost as xgb
from bench_utils import *

N_PERF_RUNS = 5
DTYPE=np.float32

xgb_params = {
    'alpha':                        0.9,
    'max_bin':                      256,
    'scale_pos_weight':             2,
    'learning_rate':                0.1,
    'subsample':                    1,
    'reg_lambda':                   1,
    "min_child_weight":             0,
    'max_depth':                    8,
    'max_leaves':                   2**8,
}

def xbg_fit():
    global model_xgb
    dtrain = xgb.DMatrix(x_train, label=y_train)
    model_xgb = xgb.train(xgb_params, dtrain, xgb_params['n_estimators'])

def xgb_predict_of_train_data():
    global result_predict_xgb_train
    dtest = xgb.DMatrix(x_train)
    result_predict_xgb_train = model_xgb.predict(dtest)

def xgb_predict_of_test_data():
    global result_predict_xgb_test
    dtest = xgb.DMatrix(x_test)
    result_predict_xgb_test = model_xgb.predict(dtest)


def load_dataset(dataset):
    global x_train, y_train, x_test, y_test

    try:
        os.mkdir(DATASET_DIR)
    except:
        pass

    datasets_dict = {
        'higgs1m': load_higgs1m,
        'msrank-10k': load_msrank_10k,
        'airline-ohe':load_airline_one_hot
    }

    x_train, y_train, x_test, y_test, n_classes = datasets_dict[dataset](DTYPE)

    if n_classes == -1:
        xgb_params['objective'] = 'reg:squarederror'
    elif n_classes == 2:
        xgb_params['objective'] = 'binary:logistic'
    else:
        xgb_params['objective'] = 'multi:softprob'
        xgb_params['num_class'] = n_classes

def parse_args():
    global N_PERF_RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', required=False, type=int, default=1000)
    parser.add_argument('--n_runs', default=N_PERF_RUNS, required=False, type=int)
    parser.add_argument('--hw', choices=['cpu', 'gpu'], metavar='stage', required=False, default='cpu')
    parser.add_argument('--log', metavar='stage', required=False, type=bool, default=False)
    parser.add_argument('--dataset', choices=['higgs1m', "airline-ohe", "msrank-10k"],
            metavar='stage', required=True)

    args = parser.parse_args()
    N_PERF_RUNS = args.n_runs

    xgb_params['n_estimators'] = args.n_iter

    if args.log:
        xgb_params['verbosity'] = 3
    else:
         xgb_params['silent'] = 1

    if args.hw == "cpu":
        xgb_params['tree_method'] = 'hist'
        xgb_params['predictor']   = 'cpu_predictor'
    elif args.hw == "gpu":
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['predictor']   = 'gpu_predictor'

    load_dataset(args.dataset)


def main():
    parse_args()

    print("Running ...")
    measure(xbg_fit,                   "XGBOOST training            ", N_PERF_RUNS)
    measure(xgb_predict_of_train_data, "XGBOOST predict (train data)", N_PERF_RUNS)
    measure(xgb_predict_of_test_data,  "XGBOOST predict (test data) ", N_PERF_RUNS)

    print("Compute quality metrics...")

    train_loglos = compute_logloss(y_train, result_predict_xgb_train)
    test_loglos = compute_logloss(y_test, result_predict_xgb_test)

    print("LogLoss for train data set = {:.6f}".format(train_loglos))
    print("LogLoss for test  data set = {:.6f}".format(test_loglos))

if __name__ == '__main__':
    main()
