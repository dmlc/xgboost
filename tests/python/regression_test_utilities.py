from __future__ import print_function

import glob
import itertools as it
import numpy as np
import os
import sys
import xgboost as xgb

try:
    from sklearn import datasets
    from sklearn.preprocessing import scale
except ImportError:
    None


class Dataset:
    def __init__(self, name, get_dataset, objective, metric,
                 has_weights=False, use_external_memory=False):
        self.name = name
        self.objective = objective
        self.metric = metric
        if has_weights:
            self.X, self.y, self.w = get_dataset()
        else:
            self.X, self.y = get_dataset()
            self.w = None
        self.use_external_memory = use_external_memory


def get_boston():
    data = datasets.load_boston()
    return data.data, data.target


def get_digits():
    data = datasets.load_digits()
    return data.data, data.target


def get_cancer():
    data = datasets.load_breast_cancer()
    return data.data, data.target


def get_sparse():
    rng = np.random.RandomState(199)
    n = 5000
    sparsity = 0.75
    X, y = datasets.make_regression(n, random_state=rng)
    X = np.array([[0.0 if rng.uniform(0, 1) < sparsity else x for x in x_row] for x_row in X])
    from scipy import sparse
    X = sparse.csr_matrix(X)
    return X, y


def get_sparse_weights():
    rng = np.random.RandomState(199)
    n = 10000
    sparsity = 0.25
    X, y = datasets.make_regression(n, random_state=rng)
    X = np.array([[np.nan if rng.uniform(0, 1) < sparsity else x for x in x_row] for x_row in X])
    w = np.array([rng.uniform(1, 10) for i in range(n)])
    return X, y, w


def train_dataset(dataset, param_in, num_rounds=10, scale_features=False):
    param = param_in.copy()
    param["objective"] = dataset.objective
    if dataset.objective == "multi:softmax":
        param["num_class"] = int(np.max(dataset.y) + 1)
    param["eval_metric"] = dataset.metric

    if scale_features:
        X = scale(dataset.X, with_mean=isinstance(dataset.X, np.ndarray))
    else:
        X = dataset.X

    if dataset.use_external_memory:
        np.savetxt('tmptmp_1234.csv', np.hstack((dataset.y.reshape(len(dataset.y), 1), X)),
                   delimiter=',')
        dtrain = xgb.DMatrix('tmptmp_1234.csv?format=csv&label_column=0#tmptmp_',
                             weight=dataset.w)
    else:
        dtrain = xgb.DMatrix(X, dataset.y, weight=dataset.w)

    print("Training on dataset: " + dataset.name, file=sys.stderr)
    print("Using parameters: " + str(param), file=sys.stderr)
    res = {}
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')],
                    evals_result=res, verbose_eval=False)

    # Free the booster and dmatrix so we can delete temporary files
    bst_copy = bst.copy()
    del bst
    del dtrain

    # Cleanup temporary files
    if dataset.use_external_memory:
        for f in glob.glob("tmptmp_*"):
            os.remove(f)

    return {"dataset": dataset, "bst": bst_copy, "param": param.copy(),
            "eval": res['train'][dataset.metric]}


def parameter_combinations(variable_param):
    """
    Enumerate all possible combinations of parameters
    """
    result = []
    names = sorted(variable_param)
    combinations = it.product(*(variable_param[Name] for Name in names))
    for set in combinations:
        param = {}
        for i, name in enumerate(names):
            param[name] = set[i]
        result.append(param)
    return result


def run_suite(param, num_rounds=10, select_datasets=None, scale_features=False):
    """
    Run the given parameters on a range of datasets. Objective and eval metric will be automatically set
    """
    datasets = [
        Dataset("Boston", get_boston, "reg:linear", "rmse"),
        Dataset("Digits", get_digits, "multi:softmax", "merror"),
        Dataset("Cancer", get_cancer, "binary:logistic", "error"),
        Dataset("Sparse regression", get_sparse, "reg:linear", "rmse"),
        Dataset("Sparse regression with weights", get_sparse_weights,
                "reg:linear", "rmse", has_weights=True),
        Dataset("Boston External Memory", get_boston, "reg:linear", "rmse",
                use_external_memory=True)
    ]

    results = [
    ]
    for d in datasets:
        if select_datasets is None or d.name in select_datasets:
            results.append(
                train_dataset(d, param, num_rounds=num_rounds, scale_features=scale_features))
    return results


def non_increasing(L, tolerance):
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))


def assert_results_non_increasing(results, tolerance=1e-5):
    for r in results:
        assert non_increasing(r['eval'], tolerance), r
