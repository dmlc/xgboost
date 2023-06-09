# pylint: disable=invalid-name
"""Utilities for data generation."""
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request

import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse

import xgboost
from xgboost.data import pandas_pyarrow_mapper

joblib = pytest.importorskip("joblib")
memory = joblib.Memory("./cachedir", verbose=0)


def np_dtypes(
    n_samples: int, n_features: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Enumerate all supported dtypes from numpy."""
    import pandas as pd

    rng = np.random.RandomState(1994)
    # Integer and float.
    orig = rng.randint(low=0, high=127, size=n_samples * n_features).reshape(
        n_samples, n_features
    )
    dtypes = [
        np.int32,
        np.int64,
        np.byte,
        np.short,
        np.intc,
        np.int_,
        np.longlong,
        np.uint32,
        np.uint64,
        np.ubyte,
        np.ushort,
        np.uintc,
        np.uint,
        np.ulonglong,
        np.float16,
        np.float32,
        np.float64,
        np.half,
        np.single,
        np.double,
    ]
    for dtype in dtypes:
        X = np.array(orig, dtype=dtype)
        yield orig, X
        yield orig.tolist(), X.tolist()

    for dtype in dtypes:
        X = np.array(orig, dtype=dtype)
        df_orig = pd.DataFrame(orig)
        df = pd.DataFrame(X)
        yield df_orig, df

    # Boolean
    orig = rng.binomial(1, 0.5, size=n_samples * n_features).reshape(
        n_samples, n_features
    )
    for dtype in [np.bool_, bool]:
        X = np.array(orig, dtype=dtype)
        yield orig, X

    for dtype in [np.bool_, bool]:
        X = np.array(orig, dtype=dtype)
        df_orig = pd.DataFrame(orig)
        df = pd.DataFrame(X)
        yield df_orig, df


def pd_dtypes() -> Generator:
    """Enumerate all supported pandas extension types."""
    import pandas as pd

    # Integer
    dtypes = [
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
    ]

    Null: Union[float, None, Any] = np.nan
    orig = pd.DataFrame(
        {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=np.float32
    )
    for Null in (np.nan, None, pd.NA):
        for dtype in dtypes:
            df = pd.DataFrame(
                {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=dtype
            )
            yield orig, df

    # Float
    Null = np.nan
    dtypes = [pd.Float32Dtype(), pd.Float64Dtype()]
    orig = pd.DataFrame(
        {"f0": [1.0, 2.0, Null, 3.0], "f1": [3.0, 2.0, Null, 1.0]}, dtype=np.float32
    )
    for Null in (np.nan, None, pd.NA):
        for dtype in dtypes:
            df = pd.DataFrame(
                {"f0": [1.0, 2.0, Null, 3.0], "f1": [3.0, 2.0, Null, 1.0]}, dtype=dtype
            )
            yield orig, df
            ser_orig = orig["f0"]
            ser = df["f0"]
            assert isinstance(ser, pd.Series)
            assert isinstance(ser_orig, pd.Series)
            yield ser_orig, ser

    # Categorical
    orig = orig.astype("category")
    for Null in (np.nan, None, pd.NA):
        df = pd.DataFrame(
            {"f0": [1.0, 2.0, Null, 3.0], "f1": [3.0, 2.0, Null, 1.0]},
            dtype=pd.CategoricalDtype(),
        )
        yield orig, df

    # Boolean
    for Null in [None, pd.NA]:
        data = {"f0": [True, False, Null, True], "f1": [False, True, Null, True]}
        # pd.NA is not convertible to bool.
        orig = pd.DataFrame(data, dtype=np.bool_ if Null is None else pd.BooleanDtype())
        df = pd.DataFrame(data, dtype=pd.BooleanDtype())
        yield orig, df


def pd_arrow_dtypes() -> Generator:
    """Pandas DataFrame with pyarrow backed type."""
    import pandas as pd
    import pyarrow as pa  # pylint: disable=import-error

    # Integer
    dtypes = pandas_pyarrow_mapper
    Null: Union[float, None, Any] = np.nan
    orig = pd.DataFrame(
        {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=np.float32
    )
    # Create a dictionary-backed dataframe, enable this when the roundtrip is
    # implemented in pandas/pyarrow
    #
    # category = pd.ArrowDtype(pa.dictionary(pa.int32(), pa.int32(), ordered=True))
    # df = pd.DataFrame({"f0": [0, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=category)

    # Error:
    # >>> df.astype("category")
    #   Function 'dictionary_encode' has no kernel matching input types
    #   (array[dictionary<values=int32, indices=int32, ordered=0>])

    # Error:
    # pd_cat_df = pd.DataFrame(
    #     {"f0": [0, 2, Null, 3], "f1": [4, 3, Null, 1]},
    #     dtype="category"
    # )
    # pa_catcodes = (
    #     df["f1"].array.__arrow_array__().combine_chunks().to_pandas().cat.codes
    # )
    # pd_catcodes = pd_cat_df["f1"].cat.codes
    # assert pd_catcodes.equals(pa_catcodes)

    for Null in (None, pd.NA):
        for dtype in dtypes:
            if dtype.startswith("float16") or dtype.startswith("bool"):
                continue
            df = pd.DataFrame(
                {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=dtype
            )
            yield orig, df

    orig = pd.DataFrame(
        {"f0": [True, False, pd.NA, True], "f1": [False, True, pd.NA, True]},
        dtype=pd.BooleanDtype(),
    )
    df = pd.DataFrame(
        {"f0": [True, False, pd.NA, True], "f1": [False, True, pd.NA, True]},
        dtype=pd.ArrowDtype(pa.bool_()),
    )
    yield orig, df


def check_inf(rng: RNG) -> None:
    """Validate there's no inf in X."""
    X = rng.random(size=32).reshape(8, 4)
    y = rng.random(size=8)
    X[5, 2] = np.inf

    with pytest.raises(ValueError, match="Input data contains `inf`"):
        xgboost.QuantileDMatrix(X, y)

    with pytest.raises(ValueError, match="Input data contains `inf`"):
        xgboost.DMatrix(X, y)


@memory.cache
def get_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the California housing dataset from sklearn."""
    datasets = pytest.importorskip("sklearn.datasets")
    data = datasets.fetch_california_housing()
    return data.data, data.target


@memory.cache
def get_digits() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the digits dataset from sklearn."""
    datasets = pytest.importorskip("sklearn.datasets")
    data = datasets.load_digits()
    return data.data, data.target


@memory.cache
def get_cancer() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the breast cancer dataset from sklearn."""
    datasets = pytest.importorskip("sklearn.datasets")
    return datasets.load_breast_cancer(return_X_y=True)


@memory.cache
def get_sparse() -> Tuple[np.ndarray, np.ndarray]:
    """Generate a sparse dataset."""
    datasets = pytest.importorskip("sklearn.datasets")
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
def get_ames_housing() -> Tuple[np.ndarray, np.ndarray]:
    """
    Number of samples: 1460
    Number of features: 20
    Number of categorical features: 10
    Number of numerical features: 10
    """
    datasets = pytest.importorskip("sklearn.datasets")
    X, y = datasets.fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

    categorical_columns_subset: List[str] = [
        "BldgType",  # 5 cats, no nan
        "GarageFinish",  # 3 cats, nan
        "LotConfig",  # 5 cats, no nan
        "Functional",  # 7 cats, no nan
        "MasVnrType",  # 4 cats, nan
        "HouseStyle",  # 8 cats, no nan
        "FireplaceQu",  # 5 cats, nan
        "ExterCond",  # 5 cats, no nan
        "ExterQual",  # 4 cats, no nan
        "PoolQC",  # 3 cats, nan
    ]

    numerical_columns_subset: List[str] = [
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
def get_mq2008(
    dpath: str,
) -> Tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
]:
    """Fetch the mq2008 dataset."""
    datasets = pytest.importorskip("sklearn.datasets")
    src = "https://s3-us-west-2.amazonaws.com/xgboost-examples/MQ2008.zip"
    target = os.path.join(dpath, "MQ2008.zip")
    if not os.path.exists(target):
        request.urlretrieve(url=src, filename=target)

    with zipfile.ZipFile(target, "r") as f:
        f.extractall(path=dpath)

    (
        x_train,
        y_train,
        qid_train,
        x_test,
        y_test,
        qid_test,
        x_valid,
        y_valid,
        qid_valid,
    ) = datasets.load_svmlight_files(
        (
            os.path.join(dpath, "MQ2008/Fold1/train.txt"),
            os.path.join(dpath, "MQ2008/Fold1/test.txt"),
            os.path.join(dpath, "MQ2008/Fold1/vali.txt"),
        ),
        query_id=True,
        zero_based=False,
    )

    return (
        x_train,
        y_train,
        qid_train,
        x_test,
        y_test,
        qid_test,
        x_valid,
        y_valid,
        qid_valid,
    )


RelData = Tuple[sparse.csr_matrix, npt.NDArray[np.int32], npt.NDArray[np.int32]]


@dataclass
class ClickFold:
    """A structure containing information about generated user-click data."""

    X: sparse.csr_matrix
    y: npt.NDArray[np.int32]
    qid: npt.NDArray[np.int32]
    score: npt.NDArray[np.float32]
    click: npt.NDArray[np.int32]
    pos: npt.NDArray[np.int64]


class RelDataCV(NamedTuple):
    """Simple data struct for holding a train-test split of a learning to rank dataset."""

    train: RelData
    test: RelData
    max_rel: int

    def is_binary(self) -> bool:
        """Whether the label consists of binary relevance degree."""
        return self.max_rel == 1


class PBM:  # pylint: disable=too-few-public-methods
    """Simulate click data with position bias model. There are other models available in
    `ULTRA <https://github.com/ULTR-Community/ULTRA.git>`_ like the cascading model.

    References
    ----------
    Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm

    """

    def __init__(self, eta: float) -> None:
        # click probability for each relevance degree. (from 0 to 4)
        self.click_prob = np.array([0.1, 0.16, 0.28, 0.52, 1.0])
        exam_prob = np.array(
            [0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06]
        )
        # Observation probability, encoding positional bias for each position
        self.exam_prob = np.power(exam_prob, eta)

    def sample_clicks_for_query(
        self, labels: npt.NDArray[np.int32], position: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int32]:
        """Sample clicks for one query based on input relevance degree and position.

        Parameters
        ----------

        labels :
            relevance_degree

        """
        labels = np.array(labels, copy=True)

        click_prob = np.zeros(labels.shape)
        # minimum
        labels[labels < 0] = 0
        # maximum
        labels[labels >= len(self.click_prob)] = -1
        click_prob = self.click_prob[labels]

        exam_prob = np.zeros(labels.shape)
        assert position.size == labels.size
        ranks = np.array(position, copy=True)
        # maximum
        ranks[ranks >= self.exam_prob.size] = -1
        exam_prob = self.exam_prob[ranks]

        rng = np.random.default_rng(1994)
        prob = rng.random(size=labels.shape[0], dtype=np.float32)

        clicks: npt.NDArray[np.int32] = np.zeros(labels.shape, dtype=np.int32)
        clicks[prob < exam_prob * click_prob] = 1
        return clicks


def rlencode(x: npt.NDArray[np.int32]) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Run length encoding using numpy, modified from:
    https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66

    """
    x = np.asarray(x)
    n = x.size
    starts = np.r_[0, np.flatnonzero(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    indptr = np.append(starts, np.array([x.size]))

    return indptr, lengths, values


def init_rank_score(
    X: sparse.csr_matrix,
    y: npt.NDArray[np.int32],
    qid: npt.NDArray[np.int32],
    sample_rate: float = 0.1,
) -> npt.NDArray[np.float32]:
    """We use XGBoost to generate the initial score instead of SVMRank for
    simplicity. Sample rate is set to 0.1 by default so that we can test with small
    datasets.

    """
    # random sample
    rng = np.random.default_rng(1994)
    n_samples = int(X.shape[0] * sample_rate)
    index = np.arange(0, X.shape[0], dtype=np.uint64)
    rng.shuffle(index)
    index = index[:n_samples]

    X_train = X[index]
    y_train = y[index]
    qid_train = qid[index]

    # Sort training data based on query id, required by XGBoost.
    sorted_idx = np.argsort(qid_train)
    X_train = X_train[sorted_idx]
    y_train = y_train[sorted_idx]
    qid_train = qid_train[sorted_idx]

    ltr = xgboost.XGBRanker(objective="rank:ndcg", tree_method="hist")
    ltr.fit(X_train, y_train, qid=qid_train)

    # Use the original order of the data.
    scores = ltr.predict(X)
    return scores


def simulate_one_fold(
    fold: Tuple[sparse.csr_matrix, npt.NDArray[np.int32], npt.NDArray[np.int32]],
    scores_fold: npt.NDArray[np.float32],
) -> ClickFold:
    """Simulate clicks for one fold."""
    X_fold, y_fold, qid_fold = fold
    assert qid_fold.dtype == np.int32

    qids = np.unique(qid_fold)

    position = np.empty((y_fold.size,), dtype=np.int64)
    clicks = np.empty((y_fold.size,), dtype=np.int32)
    pbm = PBM(eta=1.0)

    # Avoid grouping by qid as we want to preserve the original data partition by
    # the dataset authors.
    for q in qids:
        qid_mask = q == qid_fold
        qid_mask = qid_mask.reshape(qid_mask.shape[0])
        query_scores = scores_fold[qid_mask]
        # Initial rank list, scores sorted to decreasing order
        query_position = np.argsort(query_scores)[::-1]
        position[qid_mask] = query_position
        # get labels
        relevance_degrees = y_fold[qid_mask]
        query_clicks = pbm.sample_clicks_for_query(relevance_degrees, query_position)
        clicks[qid_mask] = query_clicks

    assert X_fold.shape[0] == qid_fold.shape[0], (X_fold.shape, qid_fold.shape)
    assert X_fold.shape[0] == clicks.shape[0], (X_fold.shape, clicks.shape)

    return ClickFold(X_fold, y_fold, qid_fold, scores_fold, clicks, position)


# pylint: disable=too-many-locals
def simulate_clicks(cv_data: RelDataCV) -> Tuple[ClickFold, Optional[ClickFold]]:
    """Simulate click data using position biased model (PBM)."""
    X, y, qid = list(zip(cv_data.train, cv_data.test))

    # ptr to train-test split
    indptr = np.array([0] + [v.shape[0] for v in X])
    indptr = np.cumsum(indptr)

    assert len(indptr) == 2 + 1  # train, test
    X_full = sparse.vstack(X)
    y_full = np.concatenate(y)
    qid_full = np.concatenate(qid)

    # Obtain initial relevance score for click simulation
    scores_full = init_rank_score(X_full, y_full, qid_full)
    # partition it back to (train, test) tuple
    scores = [scores_full[indptr[i - 1] : indptr[i]] for i in range(1, indptr.size)]

    X_lst, y_lst, q_lst, s_lst, c_lst, p_lst = [], [], [], [], [], []
    for i in range(indptr.size - 1):
        fold = simulate_one_fold((X[i], y[i], qid[i]), scores[i])
        X_lst.append(fold.X)
        y_lst.append(fold.y)
        q_lst.append(fold.qid)
        s_lst.append(fold.score)
        c_lst.append(fold.click)
        p_lst.append(fold.pos)

    scores_check_1 = [s_lst[i] for i in range(indptr.size - 1)]
    for i in range(2):
        assert (scores_check_1[i] == scores[i]).all()

    if len(X_lst) == 1:
        train = ClickFold(X_lst[0], y_lst[0], q_lst[0], s_lst[0], c_lst[0], p_lst[0])
        test = None
    else:
        train, test = (
            ClickFold(X_lst[i], y_lst[i], q_lst[i], s_lst[i], c_lst[i], p_lst[i])
            for i in range(len(X_lst))
        )
    return train, test


def sort_ltr_samples(
    X: sparse.csr_matrix,
    y: npt.NDArray[np.int32],
    qid: npt.NDArray[np.int32],
    clicks: npt.NDArray[np.int32],
    pos: npt.NDArray[np.int64],
) -> Tuple[
    sparse.csr_matrix,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Sort data based on query index and position."""
    sorted_idx = np.argsort(qid)
    X = X[sorted_idx]
    clicks = clicks[sorted_idx]
    qid = qid[sorted_idx]
    pos = pos[sorted_idx]

    indptr, _, _ = rlencode(qid)

    for i in range(1, indptr.size):
        beg = indptr[i - 1]
        end = indptr[i]

        assert beg < end, (beg, end)
        assert np.unique(qid[beg:end]).size == 1, (beg, end)

        query_pos = pos[beg:end]
        assert query_pos.min() == 0, query_pos.min()
        assert query_pos.max() >= query_pos.size - 1, (
            query_pos.max(),
            query_pos.size,
            i,
            np.unique(qid[beg:end]),
        )
        sorted_idx = np.argsort(query_pos)

        X[beg:end] = X[beg:end][sorted_idx]
        clicks[beg:end] = clicks[beg:end][sorted_idx]
        y[beg:end] = y[beg:end][sorted_idx]
        # not necessary
        qid[beg:end] = qid[beg:end][sorted_idx]

    data = X, clicks, y, qid

    return data
