import itertools as it
import unittest

import numpy as np
import xgboost as xgb

try:
    from sklearn import datasets
except ImportError:
    None


class Dataset:
    def __init__(self, name, dmatrix, objective, metric, cleanup_function=None):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.dmatrix = dmatrix
        self.cleanup_function = cleanup_function

    def __del__(self):
        if self.cleanup_function is not None:
            self.cleanup_function()


def get_boston():
    data = datasets.load_boston()
    return xgb.DMatrix(data.data, label=data.target)


def get_digits():
    data = datasets.load_digits()
    return xgb.DMatrix(data.data, label=data.target)


def get_cancer():
    data = datasets.load_breast_cancer()
    return xgb.DMatrix(data.data, label=data.target)


def get_digits():
    data = datasets.load_digits()
    return xgb.DMatrix(data.data, label=data.target)


def get_sparse():
    rng = np.random.RandomState(199)
    n = 5000
    sparsity = 0.75
    X, y = datasets.make_regression(n, random_state=rng)
    X = np.array([[np.nan if rng.uniform(0, 1) < sparsity else x for x in x_row] for x_row in X])
    return xgb.DMatrix(X, label=y)


def get_boston_external():
    data = datasets.load_boston()
    X = data.data
    y = data.target
    np.savetxt('tmptmp_1234.csv', np.hstack((y.reshape(len(y), 1), X)),
               delimiter=',', fmt='%10.9f')
    return xgb.DMatrix('tmptmp_1234.csv?format=csv&label_column=0#tmptmp_')


def cleanup_boston_external():
    for f in glob.glob("tmptmp_*"):
        os.remove(f)


datasets = [
    Dataset("Boston", get_boston(), "reg:linear", "rmse"),
    Dataset("Digits", get_digits(), "multi:softmax", "merror"),
    Dataset("Cancer", get_cancer(), "binary:logistic", "error"),
    Dataset("Sparse regression", get_sparse(), "reg:linear", "rmse"),
    Dataset("Boston External Memory", get_boston_external(), "reg:linear", "rmse"),
]


def train_dataset(dataset, param_in, num_rounds=10):
    param = param_in.copy()
    param["objective"] = dataset.objective
    if dataset.objective == "multi:softmax":
        param["num_class"] = int(np.max(dataset.dmatrix.get_label()) + 1)
    param["eval_metric"] = dataset.metric

    res = {}
    bst = xgb.train(param, dataset.dmatrix, num_rounds, [(dataset.dmatrix, 'train')],
                    evals_result=res)
    return {"dataset": dataset, "bst": bst, "param": param, "eval": res['train'][dataset.metric]}


def run_suite(param):
    for d in datasets:
        train_dataset(d, param)


param = {"updater": "grow_histmaker"}
run_suite(param)
param = {"updater": "grow_colmaker"}
run_suite(param)
