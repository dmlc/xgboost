# coding: utf-8
import os
import urllib
import zipfile
import sys
from contextlib import contextmanager
from io import StringIO
from xgboost.compat import SKLEARN_INSTALLED, PANDAS_INSTALLED
from xgboost.compat import DASK_INSTALLED
import pytest
import gc
import xgboost as xgb
import numpy as np
import platform

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


def skip_s390x():
    condition = platform.machine() == "s390x"
    reason = "Known to fail on s390x"
    return {"condition": condition, "reason": reason}


class IteratorForTest(xgb.core.DataIter):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.it = 0
        super().__init__("./")

    def next(self, input_data):
        if self.it == len(self.X):
            return 0
        # Use copy to make sure the iterator doesn't hold a reference to the data.
        input_data(data=self.X[self.it].copy(), label=self.y[self.it].copy())
        gc.collect()            # clear up the copy, see if XGBoost access freed memory.
        self.it += 1
        return 1

    def reset(self):
        self.it = 0

    def as_arrays(self):
        X = np.concatenate(self.X, axis=0)
        y = np.concatenate(self.y, axis=0)
        return X, y


# Contains a dataset in numpy format as well as the relevant objective and metric
class TestDataset:
    def __init__(self, name, get_dataset, objective, metric):
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
        n_samples = self.X.shape[0]
        n_batches = 10
        per_batch = n_samples // n_batches + 1

        predictor = []
        response = []
        for i in range(n_batches):
            beg = i * per_batch
            end = min((i + 1) * per_batch, n_samples)
            assert end != beg
            X = self.X[beg: end, ...]
            y = self.y[beg: end]
            predictor.append(X)
            response.append(y)

        it = IteratorForTest(predictor, response)
        return xgb.DMatrix(it)

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


@memory.cache
def get_mq2008(dpath):
    from sklearn.datasets import load_svmlight_files

    src = 'https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.zip'
    target = dpath + '/MQ2008.zip'
    if not os.path.exists(target):
        urllib.request.urlretrieve(url=src, filename=target)

    with zipfile.ZipFile(target, 'r') as f:
        f.extractall(path=dpath)

    (x_train, y_train, qid_train, x_test, y_test, qid_test,
     x_valid, y_valid, qid_valid) = load_svmlight_files(
         (dpath + "MQ2008/Fold1/train.txt",
          dpath + "MQ2008/Fold1/test.txt",
          dpath + "MQ2008/Fold1/vali.txt"),
         query_id=True, zero_based=False)

    return (x_train, y_train, qid_train, x_test, y_test, qid_test,
            x_valid, y_valid, qid_valid)


@memory.cache
def make_categorical(
    n_samples: int, n_features: int, n_categories: int, onehot: bool
):
    import pandas as pd

    rng = np.random.RandomState(1994)

    pd_dict = {}
    for i in range(n_features + 1):
        c = rng.randint(low=0, high=n_categories, size=n_samples)
        pd_dict[str(i)] = pd.Series(c, dtype=np.int64)

    df = pd.DataFrame(pd_dict)
    label = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    for i in range(0, n_features):
        label += df.iloc[:, i]
    label += 1

    df = df.astype("category")
    categories = np.arange(0, n_categories)
    for col in df.columns:
        df[col] = df[col].cat.set_categories(categories)

    if onehot:
        return pd.get_dummies(df), label
    return df, label


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
    if predt.size == 0:
        return "CustomErr", 0
    r[gt] = 1 - label[gt]
    le = predt <= 0.5
    r[le] = label[le]
    return 'CustomErr', np.sum(r)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)


def softprob_obj(classes):
    def objective(labels, predt):
        rows = labels.shape[0]
        grad = np.zeros((rows, classes), dtype=float)
        hess = np.zeros((rows, classes), dtype=float)
        eps = 1e-6
        for r in range(predt.shape[0]):
            target = labels[r]
            p = softmax(predt[r, :])
            for c in range(predt.shape[1]):
                assert target >= 0 or target <= classes
                g = p[c] - 1.0 if c == target else p[c]
                h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
                grad[r, c] = g
                hess[r, c] = h

        grad = grad.reshape((rows * classes, 1))
        hess = hess.reshape((rows * classes, 1))
        return grad, hess

    return objective


class DirectoryExcursion:
    def __init__(self, path: os.PathLike, cleanup=False):
        '''Change directory.  Change back and optionally cleaning up the directory when exit.

        '''
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))
        self.cleanup = cleanup
        self.files = {}

    def __enter__(self):
        os.chdir(self.path)
        if self.cleanup:
            self.files = {
                os.path.join(root, f)
                for root, subdir, files in os.walk(self.path) for f in files
            }

    def __exit__(self, *args):
        os.chdir(self.curdir)
        if self.cleanup:
            files = {
                os.path.join(root, f)
                for root, subdir, files in os.walk(self.path) for f in files
            }
            diff = files.difference(self.files)
            for f in diff:
                os.remove(f)


@contextmanager
def captured_output():
    """Reassign stdout temporarily in order to test printed statements
    Taken from:
    https://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python

    Also works for pytest.

    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


try:
    # Python 3.7+
    from contextlib import nullcontext as noop_context
except ImportError:
    # Python 3.6
    from contextlib import suppress as noop_context


CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.normpath(
    os.path.join(CURDIR, os.path.pardir, os.path.pardir))
