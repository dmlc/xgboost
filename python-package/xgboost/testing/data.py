# pylint: disable=invalid-name, too-many-lines
"""Utilities for data generation."""
import gc
import multiprocessing
import os
import string
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from urllib import request

import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse

from ..core import DataIter, DMatrix, QuantileDMatrix
from ..data import is_pd_cat_dtype, pandas_pyarrow_mapper
from ..sklearn import ArrayLike, XGBRanker
from ..training import train as train_fn

if TYPE_CHECKING:
    from pandas import DataFrame as DataFrameT
else:
    DataFrameT = Any

joblib = pytest.importorskip("joblib")
memory = joblib.Memory("./cachedir", verbose=0)


def np_dtypes(
    n_samples: int, n_features: int
) -> Generator[Union[Tuple[np.ndarray, np.ndarray], Tuple[list, list]], None, None]:
    """Enumerate all supported dtypes from numpy."""
    pd = pytest.importorskip("pandas")

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
    for dtype1 in [np.bool_, bool]:
        X = np.array(orig, dtype=dtype1)
        yield orig, X

    for dtype2 in [np.bool_, bool]:
        X = np.array(orig, dtype=dtype2)
        df_orig = pd.DataFrame(orig)
        df = pd.DataFrame(X)
        yield df_orig, df


def pd_dtypes() -> Generator:
    """Enumerate all supported pandas extension types."""
    pd = pytest.importorskip("pandas")

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
    for c in orig.columns:
        orig[c] = orig[c].cat.rename_categories(int)
    for Null in (np.nan, None, pd.NA):
        df = pd.DataFrame(
            {"f0": [1, 2, Null, 3], "f1": [3, 2, Null, 1]},
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
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")

    # Integer
    dtypes = pandas_pyarrow_mapper
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

    for Null in (None, pd.NA, 0):
        for dtype in dtypes:
            if dtype.startswith("float16") or dtype.startswith("bool"):
                continue
            # Use np.nan is a baseline
            orig_null = Null if not pd.isna(Null) and Null == 0 else np.nan
            orig = pd.DataFrame(
                {"f0": [1, 2, orig_null, 3], "f1": [4, 3, orig_null, 1]},
                dtype=np.float32,
            )

            df = pd.DataFrame(
                {"f0": [1, 2, Null, 3], "f1": [4, 3, Null, 1]}, dtype=dtype
            )
            yield orig, df

    # If Null is `False`, then there's no missing value.
    for Null in (pd.NA, False):
        orig = pd.DataFrame(
            {"f0": [True, False, Null, True], "f1": [False, True, Null, True]},
            dtype=pd.BooleanDtype(),
        )
        df = pd.DataFrame(
            {"f0": [True, False, Null, True], "f1": [False, True, Null, True]},
            dtype=pd.ArrowDtype(pa.bool_()),
        )
        yield orig, df


def check_inf(rng: RNG) -> None:
    """Validate there's no inf in X."""
    X = rng.random(size=32).reshape(8, 4)
    y = rng.random(size=8)
    X[5, 2] = np.inf

    with pytest.raises(ValueError, match="Input data contains `inf`"):
        QuantileDMatrix(X, y)

    with pytest.raises(ValueError, match="Input data contains `inf`"):
        DMatrix(X, y)


@memory.cache
def get_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize a dataset similar to the sklearn California housing dataset.

    The real one can be obtained via:

    .. code-block::

        import sklearn.datasets

        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    """
    n_samples = 20640
    rng = np.random.default_rng(2025)

    pd = pytest.importorskip("pandas")

    def mixture_2comp(
        means: List[float], sigmas: List[float], weights: List[float]
    ) -> np.ndarray:
        l0 = rng.normal(
            size=(int(n_samples * weights[0])), loc=means[0], scale=sigmas[0]
        )
        l1 = rng.normal(size=(n_samples - l0.shape[0]), loc=means[1], scale=sigmas[1])
        return np.concatenate([l0, l1], axis=0)

    def norm(mean: float, std: float) -> np.ndarray:
        return rng.normal(loc=mean, scale=std, size=(n_samples,))

    df = pd.DataFrame(
        {
            "Longitude": mixture_2comp(
                [-118.0703597, -121.85682825],
                [0.7897320650373969, 0.7248398629412008],
                [0.60402556, 0.39597444],
            ),
            "Latitude": mixture_2comp(
                [37.84266317, 33.86030848],
                [1.0643911549736087, 0.5049274656834589],
                [0.44485062, 0.55514938],
            ),
            "MedInc": norm(mean=3.8706710029069766, std=1.8997756945748738),
            "HouseAge": norm(mean=28.639486434108527, std=12.585252725724606),
            "AveRooms": norm(mean=5.428999742190376, std=2.474113202333516),
            "AveBedrms": norm(mean=1.096675149606208, std=0.47389937625774475),
            "Population": norm(mean=1425.4767441860465, std=1132.434687757615),
            "AveOccup": norm(mean=3.0706551594363742, std=10.385797959128219),
            "MedHouseVal": norm(mean=2.068558169089147, std=1.1539282040412253),
        }
    )
    X = df[df.columns.difference(["MedHouseVal"])].to_numpy()
    y = df["MedHouseVal"].to_numpy()
    return X, y


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


# pylint: disable=too-many-statements
@memory.cache
def get_ames_housing() -> Tuple[DataFrameT, np.ndarray]:
    """Get a synthetic version of the amse housing dataset.

    The real one can be obtained via:

    .. code-block::

        from sklearn import datasets

        datasets.fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

    Number of samples: 1460
    Number of features: 20
    Number of categorical features: 10
    Number of numerical features: 10
    """
    if TYPE_CHECKING:
        import pandas as pd
    else:
        pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(1994)
    n_samples = 1460
    df = pd.DataFrame()

    def synth_cat(
        name_proba: Dict[Union[str, float], float], density: float
    ) -> pd.Series:
        n_nulls = int(n_samples * (1 - density))
        has_nan = np.abs(1.0 - density) > 1e-6 and n_nulls > 0
        if has_nan:
            sparsity = 1.0 - density
            name_proba[np.nan] = sparsity

        keys = list(name_proba.keys())
        p = list(name_proba.values())
        p[-1] += 1.0 - np.sum(p)  # Fix floating point error
        x = rng.choice(keys, size=n_samples, p=p)

        series = pd.Series(
            x,
            dtype=pd.CategoricalDtype(
                # not NA
                filter(lambda x: isinstance(x, str), keys)
            ),
        )
        return series

    df["BldgType"] = synth_cat(
        {
            "1Fam": 0.835616,
            "2fmCon": 0.078082,
            "Duplex": 0.035616,
            "Twnhs": 0.029452,
            "TwnhsE": 0.021233,
        },
        1.0,
    )
    df["GarageFinish"] = synth_cat(
        {"Unf": 0.414384, "RFn": 0.289041, "Fin": 0.241096}, 0.94452
    )
    df["LotConfig"] = synth_cat(
        {
            "Corner": 0.180137,
            "CulDSac": 0.064384,
            "FR2": 0.032192,
            "FR3": 0.002740,
        },
        1.0,
    )
    df["Functional"] = synth_cat(
        {
            "Typ": 0.931506,
            "Min2": 0.023287,
            "Min1": 0.021232,
            "Mod": 0.010273,
            "Maj1": 0.009589,
            "Maj2": 0.003424,
            "Sev": 0.000684,
        },
        1.0,
    )
    df["MasVnrType"] = synth_cat(
        {
            "None": 0.591780,
            "BrkFace": 0.304794,
            "Stone": 0.087671,
            "BrkCmn": 0.010273,
        },
        0.99452,
    )
    df["HouseStyle"] = synth_cat(
        {
            "1Story": 0.497260,
            "2Story": 0.304794,
            "1.5Fin": 0.105479,
            "SLvl": 0.044520,
            "SFoyer": 0.025342,
            "1.5Unf": 0.009589,
            "2.5Unf": 0.007534,
            "2.5Fin": 0.005479,
        },
        1.0,
    )
    df["FireplaceQu"] = synth_cat(
        {
            "Gd": 0.260273,
            "TA": 0.214383,
            "Fa": 0.022602,
            "Ex": 0.016438,
            "Po": 0.013698,
        },
        0.527397,
    )
    df["ExterCond"] = synth_cat(
        {
            "TA": 0.878082,
            "Gd": 0.1,
            "Fa": 0.019178,
            "Ex": 0.002054,
            "Po": 0.000684,
        },
        1.0,
    )
    df["ExterQual"] = synth_cat(
        {
            "TA": 0.620547,
            "Gd": 0.334246,
            "Ex": 0.035616,
            "Fa": 0.009589,
        },
        1.0,
    )
    df["PoolQC"] = synth_cat(
        {
            "Gd": 0.002054,
            "Ex": 0.001369,
            "Fa": 0.001369,
        },
        0.004794,
    )

    # We focus on the cateogircal values here, for numerical features, simple normal
    # distribution is used, which doesn't match the original data.
    def synth_num(loc: float, std: float, density: float) -> pd.Series:
        x = rng.normal(loc=loc, scale=std, size=n_samples)
        n_nulls = int(n_samples * (1 - density))
        if np.abs(1.0 - density) > 1e-6 and n_nulls > 0:
            null_idx = rng.choice(n_samples, size=n_nulls, replace=False)
            x[null_idx] = np.nan
        return pd.Series(x, dtype=np.float64)

    df["3SsnPorch"] = synth_num(3.4095890410958902, 29.31733055678188, 1.0)
    df["Fireplaces"] = synth_num(0.613013698630137, 0.6446663863122295, 1.0)
    df["BsmtHalfBath"] = synth_num(0.057534246575342465, 0.23875264627921178, 1.0)
    df["HalfBath"] = synth_num(0.38287671232876713, 0.5028853810928914, 1.0)
    df["GarageCars"] = synth_num(1.7671232876712328, 0.7473150101111095, 1.0)
    df["TotRmsAbvGrd"] = synth_num(6.517808219178082, 1.6253932905840505, 1.0)
    df["BsmtFinSF1"] = synth_num(443.6397260273973, 456.0980908409277, 1.0)
    df["BsmtFinSF2"] = synth_num(46.54931506849315, 161.31927280654173, 1.0)
    df["GrLivArea"] = synth_num(1515.463698630137, 525.4803834232025, 1.0)
    df["ScreenPorch"] = synth_num(15.060958904109588, 55.757415281874174, 1.0)

    columns = list(df.columns)
    rng.shuffle(columns)
    df = df[columns]

    # linear interaction for testing purposes.
    y = np.zeros(shape=(n_samples,))
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            y += df[c].cat.codes.astype(np.float64)
        else:
            y += df[c].values

    # Shift and scale to match the original y.
    y *= 79442.50288288662 / y.std()
    y += 180921.19589041095 - y.mean()

    return df, y


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


def make_batches(  # pylint: disable=too-many-arguments,too-many-locals
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    use_cupy: bool = False,
    *,
    vary_size: bool = False,
    random_state: int = 1994,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Make batches of dense data."""
    X = []
    y = []
    w = []
    if use_cupy:
        import cupy

        rng = cupy.random.RandomState(np.uint64(random_state))
    else:
        rng = np.random.RandomState(random_state)
    for i in range(n_batches):
        n_samples = n_samples_per_batch + i * 10 if vary_size else n_samples_per_batch
        _X = rng.randn(n_samples, n_features)
        _y = rng.randn(n_samples)
        _w = rng.uniform(low=0, high=1, size=n_samples)
        X.append(_X)
        y.append(_y)
        w.append(_w)
    return X, y, w


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


class PBM:
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
    index: npt.NDArray = np.arange(0, X.shape[0], dtype=np.uint64)
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

    ltr = XGBRanker(objective="rank:ndcg", tree_method="hist")
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


def run_base_margin_info(DType: Callable, DMatrixT: Type[DMatrix], device: str) -> None:
    """Run tests for base margin."""
    rng = np.random.default_rng()
    X = DType(rng.normal(0, 1.0, size=100).astype(np.float32).reshape(50, 2))
    if hasattr(X, "iloc"):
        y = X.iloc[:, 0]
    else:
        y = X[:, 0]
    base_margin = X
    # no error at set
    Xy = DMatrixT(X, y, base_margin=base_margin)
    # Error at train, caused by check in predictor.
    with pytest.raises(ValueError, match=r".*base_margin.*"):
        train_fn({"tree_method": "hist", "device": device}, Xy)

    if not hasattr(X, "iloc"):
        # column major matrix
        got = DType(Xy.get_base_margin().reshape(50, 2))
        assert (got == base_margin).all()

        assert base_margin.T.flags.c_contiguous is False
        assert base_margin.T.flags.f_contiguous is True
        Xy.set_info(base_margin=base_margin.T)
        got = DType(Xy.get_base_margin().reshape(2, 50))
        assert (got == base_margin.T).all()

        # Row vs col vec.
        base_margin = y
        Xy.set_base_margin(base_margin)
        bm_col = Xy.get_base_margin()
        Xy.set_base_margin(base_margin.reshape(1, base_margin.size))
        bm_row = Xy.get_base_margin()
        assert (bm_row == bm_col).all()

        # type
        base_margin = base_margin.astype(np.float64)
        Xy.set_base_margin(base_margin)
        bm_f64 = Xy.get_base_margin()
        assert (bm_f64 == bm_col).all()

        # too many dimensions
        base_margin = X.reshape(2, 5, 2, 5)
        with pytest.raises(ValueError, match=r".*base_margin.*"):
            Xy.set_base_margin(base_margin)


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


def unique_random_strings(n_strings: int, seed: int) -> List[str]:
    """Generate n unique strings."""
    name_len = 8  # hardcoded, should be more than enough
    unique_strings: Set[str] = set()
    rng = np.random.default_rng(seed)

    while len(unique_strings) < n_strings:
        random_str = "".join(
            rng.choice(list(string.ascii_letters), size=name_len, replace=True)
        )
        unique_strings.add(random_str)

    return list(unique_strings)


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches
def make_categorical(
    n_samples: int,
    n_features: int,
    n_categories: int,
    *,
    onehot: bool,
    sparsity: float = 0.0,
    cat_ratio: float = 1.0,
    shuffle: bool = False,
    random_state: int = 1994,
    cat_dtype: np.typing.DTypeLike = np.int64,
    device: str = "cpu",
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
    cat_dtype :
        The dtype for categorical features, might be string or numeric.

    Returns
    -------
    X, y
    """
    pd = pytest.importorskip("pandas")

    # Use different rngs for column and rows. We can change the `n_samples` without
    # changing the column type.
    rng = np.random.RandomState(random_state)
    row_rng = np.random.RandomState(random_state + 1)

    df = pd.DataFrame()
    for i in range(n_features):
        choice = rng.binomial(1, cat_ratio, size=1)[0]
        if choice == 1:
            if np.issubdtype(cat_dtype, np.str_):
                # we rely on using the feature index as the seed to generate the same
                # categories for multiple calls to `make_categorical`.
                categories = np.array(unique_random_strings(n_categories, i))
                c = row_rng.choice(categories, size=n_samples, replace=True)
            else:
                categories = np.arange(0, n_categories)
                c = row_rng.randint(low=0, high=n_categories, size=n_samples)

            df[str(i)] = pd.Series(c, dtype="category")
            df[str(i)] = df[str(i)].cat.set_categories(categories)
        else:
            num = row_rng.randint(low=0, high=n_categories, size=n_samples)
            df[str(i)] = pd.Series(num, dtype=num.dtype)

    label = np.zeros(shape=(n_samples,))
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            label += df[col].cat.codes
        else:
            label += df[col]
    label += 1

    if sparsity > 0.0:
        for i in range(n_features):
            index = row_rng.randint(
                low=0, high=n_samples - 1, size=int(n_samples * sparsity)
            )
            df.iloc[index, i] = np.nan
            if is_pd_cat_dtype(df.dtypes.iloc[i]):
                assert n_categories == np.unique(df.dtypes.iloc[i].categories).size

    assert df.shape[1] == n_features
    if onehot:
        df = pd.get_dummies(df)

    if shuffle:
        columns = list(df.columns)
        row_rng.shuffle(columns)
        df = df[columns]

    if device != "cpu":
        assert device in ["cuda", "gpu"]
        import cudf
        import cupy

        df = cudf.from_pandas(df)
        label = cupy.array(label)
    return df, label


class IteratorForTest(DataIter):
    """Iterator for testing streaming DMatrix. (external memory, quantile)"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        X: Sequence,
        y: Sequence,
        w: Optional[Sequence],
        *,
        cache: Optional[str],
        on_host: bool = False,
        min_cache_page_bytes: Optional[int] = None,
    ) -> None:
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.w = w
        self.it = 0
        super().__init__(
            cache_prefix=cache,
            on_host=on_host,
            min_cache_page_bytes=min_cache_page_bytes,
        )

    def next(self, input_data: Callable) -> bool:
        if self.it == len(self.X):
            return False

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
        return True

    def reset(self) -> None:
        self.it = 0

    def as_arrays(
        self,
    ) -> Tuple[Union[np.ndarray, sparse.csr_matrix], ArrayLike, Optional[ArrayLike]]:
        """Return concatenated arrays."""
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
