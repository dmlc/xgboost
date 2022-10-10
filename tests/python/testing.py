from concurrent.futures import ThreadPoolExecutor
import os
import multiprocessing
from typing import Tuple, Union, List, Sequence, Callable
import urllib
import zipfile
import sys
from typing import Optional, Dict, Any
from contextlib import contextmanager
from io import StringIO
from xgboost.compat import SKLEARN_INSTALLED, PANDAS_INSTALLED
import pytest
import gc
import xgboost as xgb
from xgboost.core import ArrayLike
import numpy as np
from scipy import sparse
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


def no_ubjson():
    reason = "ubjson is not intsalled."
    try:
        import ubjson           # noqa
        return {"condition": False, "reason": reason}
    except ImportError:
        return {"condition": True, "reason": reason}


def no_sklearn():
    return {'condition': not SKLEARN_INSTALLED,
            'reason': 'Scikit-Learn is not installed'}


def no_dask():
    try:
        import pkg_resources

        pkg_resources.get_distribution("dask")
        DASK_INSTALLED = True
    except pkg_resources.DistributionNotFound:
        DASK_INSTALLED = False
    return {"condition": not DASK_INSTALLED, "reason": "Dask is not installed"}


def no_spark():
    try:
        import pyspark          # noqa
        SPARK_INSTALLED = True
    except ImportError:
        SPARK_INSTALLED = False
    return {"condition": not SPARK_INSTALLED, "reason": "Spark is not installed"}


def no_pandas():
    return {'condition': not PANDAS_INSTALLED,
            'reason': 'Pandas is not installed.'}


def no_arrow():
    reason = "pyarrow is not installed"
    try:
        import pyarrow  # noqa
        return {"condition": False, "reason": reason}
    except ImportError:
        return {"condition": True, "reason": reason}


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
    def __init__(
        self,
        X: Sequence,
        y: Sequence,
        w: Optional[Sequence],
        cache: Optional[str] = "./"
    ) -> None:
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.w = w
        self.it = 0
        super().__init__(cache)

    def next(self, input_data: Callable) -> int:
        if self.it == len(self.X):
            return 0

        with pytest.raises(TypeError, match="keyword args"):
            input_data(self.X[self.it], self.y[self.it], None)

        # Use copy to make sure the iterator doesn't hold a reference to the data.
        input_data(
            data=self.X[self.it].copy(),
            label=self.y[self.it].copy(),
            weight=self.w[self.it].copy() if self.w else None,
        )
        gc.collect()  # clear up the copy, see if XGBoost access freed memory.
        self.it += 1
        return 1

    def reset(self) -> None:
        self.it = 0

    def as_arrays(
        self,
    ) -> Tuple[Union[np.ndarray, sparse.csr_matrix], ArrayLike, ArrayLike]:
        if isinstance(self.X[0], sparse.csr_matrix):
            X = sparse.vstack(self.X, format="csr")
        else:
            X = np.concatenate(self.X, axis=0)
        y = np.concatenate(self.y, axis=0)
        if self.w:
            w = np.concatenate(self.w, axis=0)
        else:
            w = None
        return X, y, w


def make_batches(
    n_samples_per_batch: int, n_features: int, n_batches: int, use_cupy: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    X = []
    y = []
    w = []
    if use_cupy:
        import cupy

        rng = cupy.random.RandomState(1994)
    else:
        rng = np.random.RandomState(1994)
    for i in range(n_batches):
        _X = rng.randn(n_samples_per_batch, n_features)
        _y = rng.randn(n_samples_per_batch)
        _w = rng.uniform(low=0, high=1, size=n_samples_per_batch)
        X.append(_X)
        y.append(_y)
        w.append(_w)
    return X, y, w


def make_batches_sparse(
    n_samples_per_batch: int, n_features: int, n_batches: int, sparsity: float
) -> Tuple[List[sparse.csr_matrix], List[np.ndarray], List[np.ndarray]]:
    X = []
    y = []
    w = []
    rng = np.random.RandomState(1994)
    for i in range(n_batches):
        _X = sparse.random(
            n_samples_per_batch,
            n_features,
            1.0 - sparsity,
            format="csr",
            dtype=np.float32,
            random_state=rng,
        )
        _y = rng.randn(n_samples_per_batch)
        _w = rng.uniform(low=0, high=1, size=n_samples_per_batch)
        X.append(_X)
        y.append(_y)
        w.append(_w)
    return X, y, w


# Contains a dataset in numpy format as well as the relevant objective and metric
class TestDataset:
    def __init__(
        self, name: str, get_dataset: Callable, objective: str, metric: str
    ) -> None:
        self.name = name
        self.objective = objective
        self.metric = metric
        self.X, self.y = get_dataset()
        self.w: Optional[np.ndarray] = None
        self.margin: Optional[np.ndarray] = None

    def set_params(self, params_in: Dict[str, Any]) -> Dict[str, Any]:
        params_in['objective'] = self.objective
        params_in['eval_metric'] = self.metric
        if self.objective == "multi:softmax":
            params_in["num_class"] = int(np.max(self.y) + 1)
        return params_in

    def get_dmat(self) -> xgb.DMatrix:
        return xgb.DMatrix(
            self.X, self.y, self.w, base_margin=self.margin, enable_categorical=True
        )

    def get_device_dmat(self) -> xgb.DeviceQuantileDMatrix:
        w = None if self.w is None else cp.array(self.w)
        X = cp.array(self.X, dtype=np.float32)
        y = cp.array(self.y, dtype=np.float32)
        return xgb.DeviceQuantileDMatrix(X, y, w, base_margin=self.margin)

    def get_external_dmat(self) -> xgb.DMatrix:
        n_samples = self.X.shape[0]
        n_batches = 10
        per_batch = n_samples // n_batches + 1

        predictor = []
        response = []
        weight = []
        for i in range(n_batches):
            beg = i * per_batch
            end = min((i + 1) * per_batch, n_samples)
            assert end != beg
            X = self.X[beg: end, ...]
            y = self.y[beg: end]
            w = self.w[beg: end] if self.w is not None else None
            predictor.append(X)
            response.append(y)
            if w is not None:
                weight.append(w)

        it = IteratorForTest(predictor, response, weight if weight else None)
        return xgb.DMatrix(it)

    def __repr__(self) -> str:
        return self.name


@memory.cache
def get_california_housing():
    data = datasets.fetch_california_housing()
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
def get_ames_housing():
    """
    Number of samples: 1460
    Number of features: 20
    Number of categorical features: 10
    Number of numerical features: 10
    """
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

    categorical_columns_subset: list[str] = [
        "BldgType",             # 5 cats, no nan
        "GarageFinish",         # 3 cats, nan
        "LotConfig",            # 5 cats, no nan
        "Functional",           # 7 cats, no nan
        "MasVnrType",           # 4 cats, nan
        "HouseStyle",           # 8 cats, no nan
        "FireplaceQu",          # 5 cats, nan
        "ExterCond",            # 5 cats, no nan
        "ExterQual",            # 4 cats, no nan
        "PoolQC",               # 3 cats, nan
    ]

    numerical_columns_subset: list[str] = [
        "3SsnPorch",
        "Fireplaces",
        "BsmtHalfBath",
        "HalfBath",
        "GarageCars",
        "TotRmsAbvGrd",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "GrLivArea",
        "ScreenPorch",
    ]

    X = X[categorical_columns_subset + numerical_columns_subset]
    X[categorical_columns_subset] = X[categorical_columns_subset].astype("category")
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
    n_samples: int, n_features: int, n_categories: int, onehot: bool, sparsity=0.0,
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

    if sparsity > 0.0:
        for i in range(n_features):
            index = rng.randint(low=0, high=n_samples-1, size=int(n_samples * sparsity))
            df.iloc[index, i] = np.NaN
            assert n_categories == np.unique(df.dtypes[i].categories).size

    if onehot:
        return pd.get_dummies(df), label
    return df, label


def _cat_sampled_from():
    @strategies.composite
    def _make_cat(draw):
        n_samples = draw(strategies.integers(2, 512))
        n_features = draw(strategies.integers(1, 4))
        n_cats = draw(strategies.integers(1, 128))
        sparsity = draw(
            strategies.floats(
                min_value=0,
                max_value=1,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
        return n_samples, n_features, n_cats, sparsity

    def _build(args):
        n_samples = args[0]
        n_features = args[1]
        n_cats = args[2]
        sparsity = args[3]
        return TestDataset(
            f"{n_samples}x{n_features}-{n_cats}-{sparsity}",
            lambda: make_categorical(n_samples, n_features, n_cats, False, sparsity),
            "reg:squarederror",
            "rmse",
        )

    return _make_cat().map(_build)


categorical_dataset_strategy = _cat_sampled_from()


@memory.cache
def make_sparse_regression(
    n_samples: int, n_features: int, sparsity: float, as_dense: bool
) -> Tuple[Union[sparse.csr_matrix], np.ndarray]:
    """Make sparse matrix.

    Parameters
    ----------

    as_dense:

      Return the matrix as np.ndarray with missing values filled by NaN

    """
    if not hasattr(np.random, "default_rng"):
        # old version of numpy on s390x
        rng = np.random.RandomState(1994)
        X = sparse.random(
            m=n_samples,
            n=n_features,
            density=1.0 - sparsity,
            random_state=rng,
            format="csr",
        )
        y = rng.normal(loc=0.0, scale=1.0, size=n_samples)
        return X, y

    # Use multi-thread to speed up the generation, convenient if you use this function
    # for benchmarking.
    n_threads = multiprocessing.cpu_count()
    n_threads = min(n_threads, n_features)

    def random_csc(t_id: int) -> sparse.csc_matrix:
        rng = np.random.default_rng(1994 * t_id)
        thread_size = n_features // n_threads
        if t_id == n_threads - 1:
            n_features_tloc = n_features - t_id * thread_size
        else:
            n_features_tloc = thread_size

        X = sparse.random(
            m=n_samples,
            n=n_features_tloc,
            density=1.0 - sparsity,
            random_state=rng,
        ).tocsc()
        y = np.zeros((n_samples, 1))

        for i in range(X.shape[1]):
            size = X.indptr[i + 1] - X.indptr[i]
            if size != 0:
                y += X[:, i].toarray() * rng.random((n_samples, 1)) * 0.2

        return X, y

    futures = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i in range(n_threads):
            futures.append(executor.submit(random_csc, i))

    X_results = []
    y_results = []
    for f in futures:
        X, y = f.result()
        X_results.append(X)
        y_results.append(y)

    assert len(y_results) == n_threads

    csr: sparse.csr_matrix = sparse.hstack(X_results, format="csr")
    y = np.asarray(y_results)
    y = y.reshape((y.shape[0], y.shape[1])).T
    y = np.sum(y, axis=1)

    assert csr.shape[0] == n_samples
    assert csr.shape[1] == n_features
    assert y.shape[0] == n_samples

    if as_dense:
        arr = csr.toarray()
        assert arr.shape[0] == n_samples
        assert arr.shape[1] == n_features
        arr[arr == 0] = np.nan
        return arr, y

    return csr, y


sparse_datasets_strategy = strategies.sampled_from(
    [
        TestDataset(
            "1e5x8-0.95-csr",
            lambda: make_sparse_regression(int(1e5), 8, 0.95, False),
            "reg:squarederror",
            "rmse",
        ),
        TestDataset(
            "1e5x8-0.5-csr",
            lambda: make_sparse_regression(int(1e5), 8, 0.5, False),
            "reg:squarederror",
            "rmse",
        ),
        TestDataset(
            "1e5x8-0.5-dense",
            lambda: make_sparse_regression(int(1e5), 8, 0.5, True),
            "reg:squarederror",
            "rmse",
        ),
        TestDataset(
            "1e5x8-0.05-csr",
            lambda: make_sparse_regression(int(1e5), 8, 0.05, False),
            "reg:squarederror",
            "rmse",
        ),
        TestDataset(
            "1e5x8-0.05-dense",
            lambda: make_sparse_regression(int(1e5), 8, 0.05, True),
            "reg:squarederror",
            "rmse",
        ),
    ]
)

_unweighted_datasets_strategy = strategies.sampled_from(
    [
        TestDataset(
            "calif_housing", get_california_housing, "reg:squarederror", "rmse"
        ),
        TestDataset(
            "calif_housing-l1", get_california_housing, "reg:absoluteerror", "mae"
        ),
        TestDataset("digits", get_digits, "multi:softmax", "mlogloss"),
        TestDataset("cancer", get_cancer, "binary:logistic", "logloss"),
        TestDataset(
            "mtreg",
            lambda: datasets.make_regression(n_samples=128, n_targets=3),
            "reg:squarederror",
            "rmse",
        ),
        TestDataset("sparse", get_sparse, "reg:squarederror", "rmse"),
        TestDataset("sparse-l1", get_sparse, "reg:absoluteerror", "mae"),
        TestDataset(
            "empty",
            lambda: (np.empty((0, 100)), np.empty(0)),
            "reg:squarederror",
            "rmse",
        ),
    ]
)


@strategies.composite
def _dataset_weight_margin(draw):
    data: TestDataset = draw(_unweighted_datasets_strategy)
    if draw(strategies.booleans()):
        data.w = draw(
            arrays(np.float64, (len(data.y)), elements=strategies.floats(0.1, 2.0))
        )
    if draw(strategies.booleans()):
        num_class = 1
        if data.objective == "multi:softmax":
            num_class = int(np.max(data.y) + 1)
        elif data.name == "mtreg":
            num_class = data.y.shape[1]

        data.margin = draw(
            arrays(
                np.float64,
                (data.y.shape[0] * num_class),
                elements=strategies.floats(0.5, 1.0),
            )
        )
        if num_class != 1:
            data.margin = data.margin.reshape(data.y.shape[0], num_class)

    return data


# A strategy for drawing from a set of example datasets
# May add random weights to the dataset
dataset_strategy = _dataset_weight_margin()


def non_increasing(L, tolerance=1e-4):
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))


def eval_error_metric(predt, dtrain: xgb.DMatrix):
    """Evaluation metric for xgb.train"""
    label = dtrain.get_label()
    r = np.zeros(predt.shape)
    gt = predt > 0.5
    if predt.size == 0:
        return "CustomErr", 0
    r[gt] = 1 - label[gt]
    le = predt <= 0.5
    r[le] = label[le]
    return 'CustomErr', np.sum(r)


def eval_error_metric_skl(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Evaluation metric that looks like metrics provided by sklearn."""
    r = np.zeros(y_score.shape)
    gt = y_score > 0.5
    r[gt] = 1 - y_true[gt]
    le = y_score <= 0.5
    r[le] = y_true[le]
    return np.sum(r)


def root_mean_square(y_true: np.ndarray, y_score: np.ndarray) -> float:
    err = y_score - y_true
    rmse = np.sqrt(np.dot(err, err) / y_score.size)
    return rmse


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
