import glob
import itertools as it
import numpy as np
import os
import sys
import xgboost as xgb
from joblib import Memory
memory = Memory('./cachedir', verbose=0)

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

    def __str__(self):
        a = 'name: {name}\nobjective:{objective}, metric:{metric}, '.format(
            name=self.name,
            objective=self.objective,
            metric=self.metric)
        b = 'external memory:{use_external_memory}\n'.format(
            use_external_memory=self.use_external_memory
        )
        return a + b

    def __repr__(self):
        return self.__str__()


@memory.cache
def get_boston():
    data = datasets.load_boston()
    return data.data, data.target


@memory.cache
def get_digits():
    data = datasets.load_digits()
    return data.data, data.target


@memory.cache
def get_cancer():
    data = datasets.load_breast_cancer()
    return data.data, data.target


@memory.cache
def get_sparse():
    rng = np.random.RandomState(199)
    n = 2000
    sparsity = 0.75
    X, y = datasets.make_regression(n, random_state=rng)
    flag = rng.binomial(1, sparsity, X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if flag[i, j]:
                X[i, j] = 0.0
    from scipy import sparse
    X = sparse.csr_matrix(X)
    return X, y


def get_sparse_weights():
    return get_weights_regression(1, 10)


def get_small_weights():
    return get_weights_regression(1e-6, 1e-5)


@memory.cache
def get_weights_regression(min_weight, max_weight):
    rng = np.random.RandomState(199)
    n = 2000
    sparsity = 0.25
    X, y = datasets.make_regression(n, random_state=rng)
    flag = rng.binomial(1, sparsity, X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if flag[i, j]:
                X[i, j] = np.nan
    w = rng.uniform(min_weight, max_weight, n)
    return X, y, w


def train_dataset(dataset, param_in, num_rounds=10, scale_features=False, DMatrixT=xgb.DMatrix,
                  dmatrix_params={}):
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
        dtrain = DMatrixT('tmptmp_1234.csv?format=csv&label_column=0#tmptmp_',
                          weight=dataset.w)
    elif DMatrixT is xgb.DeviceQuantileDMatrix:
        import cupy as cp
        dtrain = DMatrixT(cp.array(X), cp.array(dataset.y),
                          weight=None if dataset.w is None else cp.array(dataset.w),
                          **dmatrix_params)
    else:
        dtrain = DMatrixT(X, dataset.y, weight=dataset.w, **dmatrix_params)

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


def run_suite(param, num_rounds=10, select_datasets=None, scale_features=False,
              DMatrixT=xgb.DMatrix, dmatrix_params={}):
    """
    Run the given parameters on a range of datasets. Objective and eval metric will be
    automatically set
    """
    datasets = [
        Dataset("Boston", get_boston, "reg:squarederror", "rmse"),
        Dataset("Digits", get_digits, "multi:softmax", "mlogloss"),
        Dataset("Cancer", get_cancer, "binary:logistic", "logloss"),
        Dataset("Sparse regression", get_sparse, "reg:squarederror", "rmse"),
        Dataset("Sparse regression with weights", get_sparse_weights,
                "reg:squarederror", "rmse", has_weights=True),
        Dataset("Small weights regression", get_small_weights,
                "reg:squarederror", "rmse", has_weights=True),
        Dataset("Boston External Memory", get_boston,
                "reg:squarederror", "rmse",
                use_external_memory=True)
    ]

    results = [
    ]
    for d in datasets:
        if select_datasets is None or d.name in select_datasets:
            results.append(
                train_dataset(d, param, num_rounds=num_rounds, scale_features=scale_features,
                              DMatrixT=DMatrixT, dmatrix_params=dmatrix_params))
    return results


def non_increasing(L, tolerance):
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))


def assert_results_non_increasing(results, tolerance=1e-5):
    for r in results:
        assert non_increasing(r['eval'], tolerance), r
