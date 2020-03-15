"""Run benchmark on the tree booster."""

import argparse
import ast
import time

import numpy as np
import xgboost as xgb

RNG = np.random.RandomState(1994)


def run_benchmark(args):
    """Runs the benchmark."""
    try:
        dtest = xgb.DMatrix('dtest.dm')
        dtrain = xgb.DMatrix('dtrain.dm')

        if not (dtest.num_col() == args.columns
                and dtrain.num_col() == args.columns):
            raise ValueError("Wrong cols")
        if not (dtest.num_row() == args.rows * args.test_size
                and dtrain.num_row() == args.rows * (1 - args.test_size)):
            raise ValueError("Wrong rows")
    except:
        print(f"Generating dataset: {args.rows} rows * {args.columns} columns")
        print(f"{args.test_size}/{1.0 - args.test_size} test/train split")
        tmp = time.time()
        X = RNG.rand(args.rows, args.columns)
        y = RNG.randint(0, 2, args.rows)
        if 0.0 < args.sparsity < 1.0:
            X = np.array([[np.nan if RNG.uniform(0, 1) < args.sparsity else x for x in x_row]
                          for x_row in X])

        train_rows = int(args.rows * (1.0 - args.test_size))
        test_rows = int(args.rows * args.test_size)
        X_train = X[:train_rows, :]
        X_test = X[-test_rows:, :]
        y_train = y[:train_rows]
        y_test = y[-test_rows:]
        print(f"Generate Time: {str(time.time() - tmp)} seconds")
        del X, y

        tmp = time.time()
        print("DMatrix Start")
        dtrain = xgb.DMatrix(X_train, y_train, nthread=-1)
        dtest = xgb.DMatrix(X_test, y_test, nthread=-1)
        print(f"DMatrix Time: {str(time.time() - tmp)} seconds")
        del X_train, y_train, X_test, y_test

        dtest.save_binary('dtest.dm')
        dtrain.save_binary('dtrain.dm')

    param = {'objective': 'binary:logistic'}
    if args.params != '':
        param.update(ast.literal_eval(args.params))

    param['tree_method'] = args.tree_method
    print(f"Training with '{param['tree_method']}'")
    tmp = time.time()
    xgb.train(param, dtrain, args.iterations, evals=[(dtest, "test")])
    print(f"Train Time: {str(time.time() - tmp)} seconds")


def main():
    """The main function.

    Defines and parses command line arguments and calls the benchmark.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_method', default='gpu_hist')
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--columns', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--params', default='',
                        help='Provide additional parameters as a Python dict string, e.g. --params '
                             '\"{\'max_depth\':2}\"')
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == '__main__':
    main()
