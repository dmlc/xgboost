# coding: utf-8
import os
import platform
from xgboost.compat import SKLEARN_INSTALLED, PANDAS_INSTALLED
from xgboost.compat import DASK_INSTALLED
import pytest
import tempfile
import xgboost as xgb
import numpy as np

hypothesis = pytest.importorskip('hypothesis')
sklearn = pytest.importorskip('sklearn')
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
from joblib import Memory
from sklearn import datasets

try:
    import cupy as cp
except ImportError:
    cp = None

memory = Memory('./cachedir', verbose=0)


def is_arm():
    return {'condition': platform.machine().lower().find('arm') != 1,
            'reason': 'Skipping expensive tests on ARM.'}

def no_sklearn():
    return {'condition': not SKLEARN_INSTALLED,
            'reason': 'Scikit-Learn is not installed'}


def no_dask():
    return {'condition': not DASK_INSTALLED,
            'reason': 'Dask is not installed'}


def no_pandas():
    return {'condition': not PANDAS_INSTALLED,
            'reason': 'Pandas is not installed.'}


def no_modin():
    reason = 'Modin is not installed.'
    try:
        import modin.pandas as _  # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}


def no_dt():
    import importlib.util
    spec = importlib.util.find_spec('datatable')
    return {'condition': spec is None,
            'reason': 'Datatable is not installed.'}


def no_matplotlib():
    reason = 'Matplotlib is not installed.'
    try:
        import matplotlib.pyplot as _  # noqa
        return {'condition': False,
                'reason': reason}
    except ImportError:
        return {'condition': True,
                'reason': reason}


def no_dask_cuda():
    reason = 'dask_cuda is not installed.'
    try:
        import dask_cuda as _  # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}


def no_cudf():
    try:
        import cudf  # noqa
        CUDF_INSTALLED = True
    except ImportError:
        CUDF_INSTALLED = False

    return {'condition': not CUDF_INSTALLED,
            'reason': 'CUDF is not installed'}


def no_cupy():
    reason = 'cupy is not installed.'
    try:
        import cupy as _  # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}


def no_dask_cudf():
    reason = 'dask_cudf is not installed.'
    try:
        import dask_cudf as _  # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}


def no_json_schema():
    reason = 'jsonschema is not installed'
    try:
        import jsonschema  # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}


def no_graphviz():
    reason = 'graphviz is not installed'
    try:
        import graphviz  # noqa
        return {'condition': False, 'reason': reason}
    except ImportError:
        return {'condition': True, 'reason': reason}


def no_multiple(*args):
    condition = False
    reason = ''
    for arg in args:
        condition = (condition or arg['condition'])
        if arg['condition']:
            reason = arg['reason']
            break
    return {'condition': condition, 'reason': reason}


# Contains a dataset in numpy format as well as the relevant objective and metric
class TestDataset:
    def __init__(self, name, get_dataset, objective, metric
                 ):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.X, self.y = get_dataset()
        self.w = None
        self.margin = None

    def set_params(self, params_in):
        params_in['objective'] = self.objective
        params_in['eval_metric'] = self.metric
        if self.objective == "multi:softmax":
            params_in["num_class"] = int(np.max(self.y) + 1)
        return params_in

    def get_dmat(self):
        return xgb.DMatrix(self.X, self.y, self.w, base_margin=self.margin)

    def get_device_dmat(self):
        w = None if self.w is None else cp.array(self.w)
        X = cp.array(self.X, dtype=np.float32)
        y = cp.array(self.y, dtype=np.float32)
        return xgb.DeviceQuantileDMatrix(X, y, w, base_margin=self.margin)

    def get_external_dmat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tmptmp_1234.csv')
            np.savetxt(path,
                       np.hstack((self.y.reshape(len(self.y), 1), self.X)),
                       delimiter=',')
            assert os.path.exists(path)
            uri = path + '?format=csv&label_column=0#tmptmp_'
            # The uri looks like:
            # 'tmptmp_1234.csv?format=csv&label_column=0#tmptmp_'
            return xgb.DMatrix(uri, weight=self.w, base_margin=self.margin)

    def __repr__(self):
        return self.name


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
                X[i, j] = np.nan
    return X, y


_unweighted_datasets_strategy = strategies.sampled_from(
    [TestDataset('boston', get_boston, 'reg:squarederror', 'rmse'),
     TestDataset('digits', get_digits, 'multi:softmax', 'mlogloss'),
     TestDataset("cancer", get_cancer, "binary:logistic", "logloss"),
     TestDataset
     ("sparse", get_sparse, "reg:squarederror", "rmse"),
     TestDataset("empty", lambda: (np.empty((0, 100)), np.empty(0)), "reg:squarederror",
                 "rmse")])


@strategies.composite
def _dataset_weight_margin(draw):
    data = draw(_unweighted_datasets_strategy)
    if draw(strategies.booleans()):
        data.w = draw(arrays(np.float64, (len(data.y)), elements=strategies.floats(0.1, 2.0)))
    if draw(strategies.booleans()):
        num_class = 1
        if data.objective == "multi:softmax":
            num_class = int(np.max(data.y) + 1)
        data.margin = draw(
            arrays(np.float64, (len(data.y) * num_class), elements=strategies.floats(0.5, 1.0)))

    return data


# A strategy for drawing from a set of example datasets
# May add random weights to the dataset
dataset_strategy = _dataset_weight_margin()


def non_increasing(L, tolerance=1e-4):
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))


def eval_error_metric(predt, dtrain: xgb.DMatrix):
    label = dtrain.get_label()
    r = np.zeros(predt.shape)
    gt = predt > 0.5
    r[gt] = 1 - label[gt]
    le = predt <= 0.5
    r[le] = label[le]
    return 'CustomErr', np.sum(r)


CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.normpath(
    os.path.join(CURDIR, os.path.pardir, os.path.pardir))
