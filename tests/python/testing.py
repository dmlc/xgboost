# coding: utf-8
from xgboost.compat import SKLEARN_INSTALLED, PANDAS_INSTALLED
from xgboost.compat import DASK_INSTALLED
from hypothesis import strategies
from hypothesis.extra.numpy import arrays
from joblib import Memory
from sklearn import datasets
import xgboost as xgb
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

memory = Memory('./cachedir', verbose=0)


def no_sklearn():
    return {'condition': not SKLEARN_INSTALLED,
            'reason': 'Scikit-Learn is not installed'}


def no_dask():
    return {'condition': not DASK_INSTALLED,
            'reason': 'Dask is not installed'}


def no_pandas():
    return {'condition': not PANDAS_INSTALLED,
            'reason': 'Pandas is not installed.'}


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


# Contains a dataset in numpy format as well as the relevant objective and metric
class TestDataset:
    def __init__(self, name, get_dataset, objective, metric
                 ):
        self.name = name
        self.objective = objective
        self.metric = metric
        self.X, self.y = get_dataset()
        self.w = None

    def set_params(self, params_in):
        params_in['objective'] = self.objective
        params_in['eval_metric'] = self.metric
        if self.objective == "multi:softmax":
            params_in["num_class"] = int(np.max(self.y) + 1)
        return params_in

    def get_dmat(self):
        return xgb.DMatrix(self.X, self.y, self.w)

    def get_device_dmat(self):
        w = None if self.w is None else cp.array(self.w)
        X = cp.array(self.X, dtype=np.float32)
        y = cp.array(self.y, dtype=np.float32)
        return xgb.DeviceQuantileDMatrix(X, y, w)

    def get_external_dmat(self):
        np.savetxt('tmptmp_1234.csv', np.hstack((self.y.reshape(len(self.y), 1), self.X)),
                   delimiter=',')
        return xgb.DMatrix('tmptmp_1234.csv?format=csv&label_column=0#tmptmp_',
                           weight=self.w)

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
    [TestDataset('boston', get_boston, 'reg:squarederror', 'logloss'),
     TestDataset('digits', get_digits, 'multi:softmax', 'mlogloss'),
     TestDataset("cancer", get_cancer, "binary:logistic", "logloss"),
     TestDataset
     ("sparse", get_sparse, "reg:squarederror", "rmse"),
     TestDataset("empty", lambda: (np.empty((0, 100)), np.empty(0)), "reg:squarederror",
                 "rmse")])


@strategies.composite
def _dataset_and_weight(draw):
    data = draw(_unweighted_datasets_strategy)
    if draw(strategies.booleans()):
        data.w = draw(arrays(np.float64, (len(data.y)), elements=strategies.floats(0.1, 2.0)))
    return data

# A strategy for drawing from a set of example datasets
# May add random weights to the dataset
dataset_strategy = _dataset_and_weight()


def non_increasing(L, tolerance=1e-4):
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))
