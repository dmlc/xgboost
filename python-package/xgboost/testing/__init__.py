"""Utilities for defining Python tests. The module is private and subject to frequent
change without notice.

"""

# pylint: disable=invalid-name,missing-function-docstring,import-error
import gc
import importlib.util
import multiprocessing
import os
import platform
import queue
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import StringIO
from platform import system
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
import pytest
from scipy import sparse

import xgboost as xgb
from xgboost import RabitTracker
from xgboost.core import ArrayLike
from xgboost.sklearn import SklObjective
from xgboost.testing.data import (
    get_california_housing,
    get_cancer,
    get_digits,
    get_sparse,
    make_batches,
    memory,
)

hypothesis = pytest.importorskip("hypothesis")

# pylint:disable=wrong-import-position,wrong-import-order
from hypothesis import strategies
from hypothesis.extra.numpy import arrays

datasets = pytest.importorskip("sklearn.datasets")

PytestSkip = TypedDict("PytestSkip", {"condition": bool, "reason": str})


def has_ipv6() -> bool:
    """Check whether IPv6 is enabled on this host."""
    # connection error in macos, still need some fixes.
    if system() not in ("Linux", "Windows"):
        return False

    if socket.has_ipv6:
        try:
            with socket.socket(
                socket.AF_INET6, socket.SOCK_STREAM
            ) as server, socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as client:
                server.bind(("::1", 0))
                port = server.getsockname()[1]
                server.listen()

                client.connect(("::1", port))
                conn, _ = server.accept()

                client.sendall("abc".encode())
                msg = conn.recv(3).decode()
                # if the code can be executed to this point, the message should be
                # correct.
                assert msg == "abc"
            return True
        except OSError:
            pass
    return False


def no_mod(name: str) -> PytestSkip:
    spec = importlib.util.find_spec(name)
    return {"condition": spec is None, "reason": f"{name} is not installed."}


def no_ipv6() -> PytestSkip:
    """PyTest skip mark for IPv6."""
    return {"condition": not has_ipv6(), "reason": "IPv6 is required to be enabled."}


def not_linux() -> PytestSkip:
    return {"condition": system() != "Linux", "reason": "Linux is required."}


def no_ubjson() -> PytestSkip:
    return no_mod("ubjson")


def no_sklearn() -> PytestSkip:
    return no_mod("sklearn")


def no_dask() -> PytestSkip:
    return no_mod("dask")


def no_dask_ml() -> PytestSkip:
    if sys.platform.startswith("win"):
        return {"reason": "Unsupported platform.", "condition": True}
    return no_mod("dask_ml")


def no_spark() -> PytestSkip:
    if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
        return {"reason": "Unsupported platform.", "condition": True}
    return no_mod("pyspark")


def no_pandas() -> PytestSkip:
    return no_mod("pandas")


def no_arrow() -> PytestSkip:
    return no_mod("pyarrow")


def no_modin() -> PytestSkip:
    return no_mod("modin")


def no_dt() -> PytestSkip:
    return no_mod("datatable")


def no_matplotlib() -> PytestSkip:
    reason = "Matplotlib is not installed."
    try:
        import matplotlib.pyplot as _  # noqa

        return {"condition": False, "reason": reason}
    except ImportError:
        return {"condition": True, "reason": reason}


def no_dask_cuda() -> PytestSkip:
    return no_mod("dask_cuda")


def no_cudf() -> PytestSkip:
    return no_mod("cudf")


def no_cupy() -> PytestSkip:
    skip_cupy = no_mod("cupy")
    if not skip_cupy["condition"] and system() == "Windows":
        import cupy as cp

        # Cupy might run into issue on Windows due to missing compiler
        try:
            cp.array([1, 2, 3]).sum()
        except Exception:  # pylint: disable=broad-except
            skip_cupy["condition"] = True
    return skip_cupy


def no_dask_cudf() -> PytestSkip:
    return no_mod("dask_cudf")


def no_json_schema() -> PytestSkip:
    return no_mod("jsonschema")


def no_graphviz() -> PytestSkip:
    return no_mod("graphviz")


def no_rmm() -> PytestSkip:
    return no_mod("rmm")


def no_multiple(*args: Any) -> PytestSkip:
    condition = False
    reason = ""
    for arg in args:
        condition = condition or arg["condition"]
        if arg["condition"]:
            reason = arg["reason"]
            break
    return {"condition": condition, "reason": reason}


def skip_win() -> PytestSkip:
    return {"reason": "Unsupported platform.", "condition": is_windows()}


class IteratorForTest(xgb.core.DataIter):
    """Iterator for testing streaming DMatrix. (external memory, quantile)"""

    def __init__(
        self,
        X: Sequence,
        y: Sequence,
        w: Optional[Sequence],
        cache: Optional[str],
    ) -> None:
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.w = w
        self.it = 0
        super().__init__(cache_prefix=cache)

    def next(self, input_data: Callable) -> int:
        if self.it == len(self.X):
            return 0

        with pytest.raises(TypeError, match="Keyword argument"):
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
    ) -> Tuple[Union[np.ndarray, sparse.csr_matrix], ArrayLike, Optional[ArrayLike]]:
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


def make_regression(
    n_samples: int, n_features: int, use_cupy: bool
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Make a simple regression dataset."""
    X, y, w = make_batches(n_samples, n_features, 1, use_cupy)
    return X[0], y[0], w[0]


def make_batches_sparse(
    n_samples_per_batch: int, n_features: int, n_batches: int, sparsity: float
) -> Tuple[List[sparse.csr_matrix], List[np.ndarray], List[np.ndarray]]:
    X = []
    y = []
    w = []
    rng = np.random.RandomState(1994)
    for _ in range(n_batches):
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


class TestDataset:
    """Contains a dataset in numpy format as well as the relevant objective and metric."""

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
        params_in["objective"] = self.objective
        params_in["eval_metric"] = self.metric
        if self.objective == "multi:softmax":
            params_in["num_class"] = int(np.max(self.y) + 1)
        return params_in

    def get_dmat(self) -> xgb.DMatrix:
        return xgb.DMatrix(
            self.X,
            self.y,
            weight=self.w,
            base_margin=self.margin,
            enable_categorical=True,
        )

    def get_device_dmat(self, max_bin: Optional[int]) -> xgb.QuantileDMatrix:
        import cupy as cp

        w = None if self.w is None else cp.array(self.w)
        X = cp.array(self.X, dtype=np.float32)
        y = cp.array(self.y, dtype=np.float32)
        return xgb.QuantileDMatrix(
            X, y, weight=w, base_margin=self.margin, max_bin=max_bin
        )

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
            X = self.X[beg:end, ...]
            y = self.y[beg:end]
            w = self.w[beg:end] if self.w is not None else None
            predictor.append(X)
            response.append(y)
            if w is not None:
                weight.append(w)

        it = IteratorForTest(
            predictor, response, weight if weight else None, cache="cache"
        )
        return xgb.DMatrix(it)

    def __repr__(self) -> str:
        return self.name


# pylint: disable=too-many-arguments,too-many-locals
@memory.cache
def make_categorical(
    n_samples: int,
    n_features: int,
    n_categories: int,
    onehot: bool,
    sparsity: float = 0.0,
    cat_ratio: float = 1.0,
    shuffle: bool = False,
) -> Tuple[ArrayLike, np.ndarray]:
    """Generate categorical features for test.

    Parameters
    ----------
    n_categories:
        Number of categories for categorical features.
    onehot:
        Should we apply one-hot encoding to the data?
    sparsity:
        The ratio of the amount of missing values over the number of all entries.
    cat_ratio:
        The ratio of features that are categorical.
    shuffle:
        Whether we should shuffle the columns.

    Returns
    -------
    X, y
    """
    import pandas as pd
    from pandas.api.types import is_categorical_dtype

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

    categories = np.arange(0, n_categories)
    for col in df.columns:
        if rng.binomial(1, cat_ratio, size=1)[0] == 1:
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.set_categories(categories)

    if sparsity > 0.0:
        for i in range(n_features):
            index = rng.randint(
                low=0, high=n_samples - 1, size=int(n_samples * sparsity)
            )
            df.iloc[index, i] = np.nan
            if is_categorical_dtype(df.dtypes[i]):
                assert n_categories == np.unique(df.dtypes[i].categories).size

    if onehot:
        df = pd.get_dummies(df)

    if shuffle:
        columns = list(df.columns)
        rng.shuffle(columns)
        df = df[columns]

    return df, label


def make_ltr(
    n_samples: int, n_features: int, n_query_groups: int, max_rel: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a dataset for testing LTR."""
    rng = np.random.default_rng(1994)
    X = rng.normal(0, 1.0, size=n_samples * n_features).reshape(n_samples, n_features)
    y = np.sum(X, axis=1)
    y -= y.min()
    y = np.round(y / y.max() * max_rel).astype(np.int32)

    qid = rng.integers(0, n_query_groups, size=n_samples, dtype=np.int32)
    w = rng.normal(0, 1.0, size=n_query_groups)
    w -= np.min(w)
    w /= np.max(w)
    qid = np.sort(qid)
    return X, y, qid, w


def _cat_sampled_from() -> strategies.SearchStrategy:
    @strategies.composite
    def _make_cat(draw: Callable) -> Tuple[int, int, int, float]:
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

    def _build(args: Tuple[int, int, int, float]) -> TestDataset:
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

    return _make_cat().map(_build)  # pylint: disable=no-member


categorical_dataset_strategy: strategies.SearchStrategy = _cat_sampled_from()


# pylint: disable=too-many-locals
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
    n_threads = min(multiprocessing.cpu_count(), n_features)

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


def make_datasets_with_margin(
    unweighted_strategy: strategies.SearchStrategy,
) -> Callable[[], strategies.SearchStrategy[TestDataset]]:
    """Factory function for creating strategies that generates datasets with weight and
    base margin.

    """

    @strategies.composite
    def weight_margin(draw: Callable) -> TestDataset:
        data: TestDataset = draw(unweighted_strategy)
        if draw(strategies.booleans()):
            data.w = draw(
                arrays(np.float64, (len(data.y)), elements=strategies.floats(0.1, 2.0))
            )
        if draw(strategies.booleans()):
            num_class = 1
            if data.objective == "multi:softmax":
                num_class = int(np.max(data.y) + 1)
            elif data.name.startswith("mtreg"):
                num_class = data.y.shape[1]

            data.margin = draw(
                arrays(
                    np.float64,
                    (data.y.shape[0] * num_class),
                    elements=strategies.floats(0.5, 1.0),
                )
            )
            assert data.margin is not None
            if num_class != 1:
                data.margin = data.margin.reshape(data.y.shape[0], num_class)

        return data

    return weight_margin


# A strategy for drawing from a set of example datasets. May add random weights to the
# dataset
def make_dataset_strategy() -> strategies.SearchStrategy[TestDataset]:
    _unweighted_datasets_strategy = strategies.sampled_from(
        [
            TestDataset(
                "calif_housing", get_california_housing, "reg:squarederror", "rmse"
            ),
            TestDataset(
                "calif_housing-l1", get_california_housing, "reg:absoluteerror", "mae"
            ),
            TestDataset("cancer", get_cancer, "binary:logistic", "logloss"),
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
    return make_datasets_with_margin(_unweighted_datasets_strategy)()


_unweighted_multi_datasets_strategy = strategies.sampled_from(
    [
        TestDataset("digits", get_digits, "multi:softmax", "mlogloss"),
        TestDataset(
            "mtreg",
            lambda: datasets.make_regression(n_samples=128, n_features=2, n_targets=3),
            "reg:squarederror",
            "rmse",
        ),
        TestDataset(
            "mtreg-l1",
            lambda: datasets.make_regression(n_samples=128, n_features=2, n_targets=3),
            "reg:absoluteerror",
            "mae",
        ),
    ]
)

# A strategy for drawing from a set of multi-target/multi-class datasets.
multi_dataset_strategy = make_datasets_with_margin(
    _unweighted_multi_datasets_strategy
)()


def non_increasing(L: Sequence[float], tolerance: float = 1e-4) -> bool:
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))


def predictor_equal(lhs: xgb.DMatrix, rhs: xgb.DMatrix) -> bool:
    """Assert whether two DMatrices contain the same predictors."""
    lcsr = lhs.get_data()
    rcsr = rhs.get_data()
    return all(
        (
            np.array_equal(lcsr.data, rcsr.data),
            np.array_equal(lcsr.indices, rcsr.indices),
            np.array_equal(lcsr.indptr, rcsr.indptr),
        )
    )


M = TypeVar("M", xgb.Booster, xgb.XGBModel)


def eval_error_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, np.float64]:
    """Evaluation metric for xgb.train"""
    label = dtrain.get_label()
    r = np.zeros(predt.shape)
    gt = predt > 0.5
    if predt.size == 0:
        return "CustomErr", np.float64(0.0)
    r[gt] = 1 - label[gt]
    le = predt <= 0.5
    r[le] = label[le]
    return "CustomErr", np.sum(r)


def eval_error_metric_skl(y_true: np.ndarray, y_score: np.ndarray) -> np.float64:
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


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x)
    return e / np.sum(e)


def softprob_obj(
    classes: int, use_cupy: bool = False, order: str = "C", gdtype: str = "float32"
) -> SklObjective:
    """Custom softprob objective for testing.

    Parameters
    ----------
    use_cupy :
        Whether the objective should return cupy arrays.
    order :
        The order of gradient matrices. "C" or "F".
    gdtype :
        DType for gradient. Hessian is not set. This is for testing asymmetric types.
    """
    if use_cupy:
        import cupy as backend
    else:
        backend = np

    def objective(
        labels: backend.ndarray, predt: backend.ndarray
    ) -> Tuple[backend.ndarray, backend.ndarray]:
        rows = labels.shape[0]
        grad = backend.zeros((rows, classes), dtype=np.float32)
        hess = backend.zeros((rows, classes), dtype=np.float32)
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

        grad = grad.reshape((rows, classes))
        hess = hess.reshape((rows, classes))
        grad = backend.require(grad, requirements=order, dtype=gdtype)
        hess = backend.require(hess, requirements=order)
        return grad, hess

    return objective


def ls_obj(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Least squared error."""
    grad = y_pred - y_true
    hess = np.ones(len(y_true))
    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight
    return grad, hess


class DirectoryExcursion:
    """Change directory.  Change back and optionally cleaning up the directory when
    exit.

    """

    def __init__(self, path: Union[os.PathLike, str], cleanup: bool = False):
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))
        self.cleanup = cleanup
        self.files: Set[str] = set()

    def __enter__(self) -> None:
        os.chdir(self.path)
        if self.cleanup:
            self.files = {
                os.path.join(root, f)
                for root, subdir, files in os.walk(os.path.expanduser(self.path))
                for f in files
            }

    def __exit__(self, *args: Any) -> None:
        os.chdir(self.curdir)
        if self.cleanup:
            files = {
                os.path.join(root, f)
                for root, subdir, files in os.walk(os.path.expanduser(self.path))
                for f in files
            }
            diff = files.difference(self.files)
            for f in diff:
                os.remove(f)


@contextmanager
def captured_output() -> Generator[Tuple[StringIO, StringIO], None, None]:
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


def timeout(sec: int, *args: Any, enable: bool = True, **kwargs: Any) -> Any:
    """Make a pytest mark for the `pytest-timeout` package.

    Parameters
    ----------
    sec :
        Timeout seconds.
    enable :
        Control whether timeout should be applied, used for debugging.

    Returns
    -------
    pytest.mark.timeout
    """

    if enable:
        return pytest.mark.timeout(sec, *args, **kwargs)
    return pytest.mark.timeout(None, *args, **kwargs)


def setup_rmm_pool(_: Any, pytestconfig: pytest.Config) -> None:
    if pytestconfig.getoption("--use-rmm-pool"):
        if no_rmm()["condition"]:
            raise ImportError("The --use-rmm-pool option requires the RMM package")
        if no_dask_cuda()["condition"]:
            raise ImportError(
                "The --use-rmm-pool option requires the dask_cuda package"
            )
        import rmm
        from dask_cuda.utils import get_n_gpus

        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=1024 * 1024 * 1024,
            devices=list(range(get_n_gpus())),
        )


def get_client_workers(client: Any) -> List[str]:
    "Get workers from a dask client."
    workers = client.scheduler_info()["workers"]
    return list(workers.keys())


def demo_dir(path: str) -> str:
    """Look for the demo directory based on the test file name."""
    path = normpath(os.path.dirname(path))
    while True:
        subdirs = [f.path for f in os.scandir(path) if f.is_dir()]
        subdirs = [os.path.basename(d) for d in subdirs]
        if "demo" in subdirs:
            return os.path.join(path, "demo")
        new_path = normpath(os.path.join(path, os.path.pardir))
        assert new_path != path
        path = new_path


def normpath(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def data_dir(path: str) -> str:
    return os.path.join(demo_dir(path), "data")


def load_agaricus(path: str) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
    dpath = data_dir(path)
    dtrain = xgb.DMatrix(os.path.join(dpath, "agaricus.txt.train?format=libsvm"))
    dtest = xgb.DMatrix(os.path.join(dpath, "agaricus.txt.test?format=libsvm"))
    return dtrain, dtest


def project_root(path: str) -> str:
    return normpath(os.path.join(demo_dir(path), os.path.pardir))


def run_with_rabit(
    world_size: int, test_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> None:
    exception_queue: queue.Queue = queue.Queue()

    def run_worker(rabit_env: Dict[str, Union[str, int]]) -> None:
        try:
            with xgb.collective.CommunicatorContext(**rabit_env):
                test_fn(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            exception_queue.put(e)

    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=world_size)
    tracker.start()

    workers = []
    for _ in range(world_size):
        worker = threading.Thread(target=run_worker, args=(tracker.worker_args(),))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
        assert exception_queue.empty(), f"Worker failed: {exception_queue.get()}"

    tracker.wait_for()


def column_split_feature_names(
    feature_names: List[Union[str, int]], world_size: int
) -> List[str]:
    """Get the global list of feature names from the local feature names."""
    return [
        f"{rank}.{feature}" for rank in range(world_size) for feature in feature_names
    ]


def is_windows() -> bool:
    """Check if the current platform is Windows."""
    return platform.system() == "Windows"
