# pylint: disable=too-many-arguments
"""Copyright 2019-2024, XGBoost contributors"""

import logging
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import dask
import distributed
import numpy as np
import pandas as pd
from dask import dataframe as dd

from .. import collective as coll
from .._typing import _T, FeatureNames
from ..compat import concat, import_cupy
from ..core import DataIter, DMatrix, QuantileDMatrix
from ..data import is_on_cuda

LOGGER = logging.getLogger("[xgboost.dask]")

_DataParts = List[Dict[str, Any]]


def dconcat(value: Sequence[_T]) -> _T:
    """Concatenate sequence of partitions."""
    try:
        return concat(value)
    except TypeError:
        return dd.multi.concat(list(value), axis=0)


meta = [
    "label",
    "weight",
    "base_margin",
    "qid",
    "label_lower_bound",
    "label_upper_bound",
]


class DaskPartitionIter(DataIter):  # pylint: disable=R0902
    """A data iterator for the `DaskQuantileDMatrix`."""

    def __init__(
        self,
        data: List[Any],
        feature_names: Optional[FeatureNames] = None,
        feature_types: Optional[Union[Any, List[Any]]] = None,
        feature_weights: Optional[Any] = None,
        **kwargs: Optional[List[Any]],
    ) -> None:
        types = (Sequence, type(None))
        # Samples
        self._data = data
        for k in meta:
            setattr(self, k, kwargs.get(k, None))
            assert isinstance(getattr(self, k), types)

        # Feature info
        self._feature_names = feature_names
        self._feature_types = feature_types
        self._feature_weights = feature_weights

        assert isinstance(self._data, Sequence)

        self._iter = 0  # set iterator to 0
        super().__init__(release_data=True)

    def _get(self, attr: str) -> Optional[Any]:
        if getattr(self, attr) is not None:
            return getattr(self, attr)[self._iter]
        return None

    def data(self) -> Any:
        """Utility function for obtaining current batch of data."""
        return self._data[self._iter]

    def reset(self) -> None:
        """Reset the iterator"""
        self._iter = 0

    def next(self, input_data: Callable) -> bool:
        """Yield next batch of data"""
        if self._iter == len(self._data):
            # Return False when there's no more batch.
            return False

        kwargs = {k: self._get(k) for k in meta}
        input_data(
            data=self.data(),
            group=None,
            feature_names=self._feature_names,
            feature_types=self._feature_types,
            feature_weights=self._feature_weights,
            **kwargs,
        )
        self._iter += 1
        return True


@overload
def _add_column(df: dd.DataFrame, col: dd.Series) -> Tuple[dd.DataFrame, str]: ...


@overload
def _add_column(df: dd.DataFrame, col: None) -> Tuple[dd.DataFrame, None]: ...


def _add_column(
    df: dd.DataFrame, col: Optional[dd.Series]
) -> Tuple[dd.DataFrame, Optional[str]]:
    if col is None:
        return df, col

    trails = 0
    uid = f"{col.name}_{trails}"
    while uid in df.columns:
        trails += 1
        uid = f"{col.name}_{trails}"

    df = df.assign(**{uid: col})
    return df, uid


def no_group_split(  # pylint: disable=too-many-positional-arguments
    device: str | None,
    df: dd.DataFrame,
    qid: dd.Series,
    y: dd.Series,
    sample_weight: Optional[dd.Series],
    base_margin: Optional[dd.Series],
) -> Tuple[
    dd.DataFrame, dd.Series, dd.Series, Optional[dd.Series], Optional[dd.Series]
]:
    """A function to prevent query group from being scattered to different
    workers. Please see the tutorial in the document for the implication for not having
    partition boundary based on query groups.

    """

    df, qid_uid = _add_column(df, qid)
    df, y_uid = _add_column(df, y)
    df, w_uid = _add_column(df, sample_weight)
    df, bm_uid = _add_column(df, base_margin)

    # `tasks` shuffle is required as of rapids 24.12
    shuffle = "p2p" if device is None or device == "cpu" else "tasks"
    with dask.config.set({"dataframe.shuffle.method": shuffle}):
        df = df.persist()
        # Encode the QID to make it dense.
        df[qid_uid] = df[qid_uid].astype("category").cat.as_known().cat.codes
        # The shuffle here is costly.
        df = df.sort_values(by=qid_uid)
        cnt = df.groupby(qid_uid)[qid_uid].count()
        div = cnt.index.compute().values.tolist()
        div = sorted(div)
        div = tuple(div + [div[-1] + 1])

        df = df.set_index(
            qid_uid,
            drop=False,
            divisions=div,
        ).persist()

    qid = df[qid_uid]
    y = df[y_uid]
    sample_weight, base_margin = (
        cast(dd.Series, df[uid]) if uid is not None else None for uid in (w_uid, bm_uid)
    )

    uids = [uid for uid in [qid_uid, y_uid, w_uid, bm_uid] if uid is not None]
    df = df.drop(uids, axis=1).persist()
    return df, qid, y, sample_weight, base_margin


def sort_data_by_qid(**kwargs: List[Any]) -> Dict[str, List[Any]]:
    """Sort worker-local data by query ID for learning to rank tasks."""
    data_parts = kwargs.get("data")
    assert data_parts is not None
    n_parts = len(data_parts)

    if is_on_cuda(data_parts[0]):
        from cudf import DataFrame
    else:
        from pandas import DataFrame

    def get_dict(i: int) -> Dict[str, list]:
        """Return a dictionary containing all the meta info and all partitions."""

        def _get(attr: Optional[List[Any]]) -> Optional[list]:
            if attr is not None:
                return attr[i]
            return None

        data_opt = {name: _get(kwargs.get(name, None)) for name in meta}
        # Filter out None values.
        data = {k: v for k, v in data_opt.items() if v is not None}
        return data

    def map_fn(i: int) -> pd.DataFrame:
        data = get_dict(i)
        return DataFrame(data)

    meta_parts = [map_fn(i) for i in range(n_parts)]
    dfq = concat(meta_parts)
    if dfq.qid.is_monotonic_increasing:
        return kwargs

    LOGGER.warning(
        "[r%d]: Sorting data with %d partitions for ranking. "
        "This is a costly operation and will increase the memory usage significantly. "
        "To avoid this warning, sort the data based on qid before passing it into "
        "XGBoost. Alternatively, you can use set the `allow_group_split` to False.",
        coll.get_rank(),
        n_parts,
    )
    # I tried to construct a new dask DF to perform the sort, but it's quite difficult
    # to get the partition alignment right. Along with the still maturing shuffle
    # implementation and GPU compatibility, a simple concat is used.
    #
    # In case it might become useful one day, I managed to get a CPU version working,
    # albeit qutie slow (much slower than concatenated sort). The implementation merges
    # everything into a single Dask DF and runs `DF.sort_values`, then retrieve the
    # individual X,y,qid, ... from calculated partition values `client.compute([p for p
    # in df.partitions])`. It was to avoid creating mismatched partitions.
    dfx = concat(data_parts)

    if is_on_cuda(dfq):
        cp = import_cupy()
        sorted_idx = cp.argsort(dfq.qid)
    else:
        sorted_idx = np.argsort(dfq.qid)
    dfq = dfq.iloc[sorted_idx, :]

    if hasattr(dfx, "iloc"):
        dfx = dfx.iloc[sorted_idx, :]
    else:
        dfx = dfx[sorted_idx, :]

    kwargs.update({"data": [dfx]})
    for i, c in enumerate(dfq.columns):
        assert c in kwargs
        kwargs.update({c: [dfq[c]]})

    return kwargs


def _get_worker_parts(list_of_parts: _DataParts) -> Dict[str, List[Any]]:
    assert isinstance(list_of_parts, list)
    result: Dict[str, List[Any]] = {}

    def append(i: int, name: str) -> None:
        if name in list_of_parts[i]:
            part = list_of_parts[i][name]
        else:
            part = None
        if part is not None:
            if name not in result:
                result[name] = []
            result[name].append(part)

    for i, _ in enumerate(list_of_parts):
        append(i, "data")
        for k in meta:
            append(i, k)

    qid = result.get("qid", None)
    if qid is not None:
        result = sort_data_by_qid(**result)
    return result


def _create_quantile_dmatrix(
    *,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[Any, List[Any]]],
    feature_weights: Optional[Any],
    missing: float,
    nthread: int,
    parts: Optional[_DataParts],
    max_bin: int,
    enable_categorical: bool,
    max_quantile_batches: Optional[int],
    ref: Optional[DMatrix] = None,
) -> QuantileDMatrix:
    worker = distributed.get_worker()
    if parts is None:
        msg = f"Worker {worker.address} has an empty DMatrix."
        LOGGER.warning(msg)

        Xy = QuantileDMatrix(
            np.empty((0, 0)),
            feature_names=feature_names,
            feature_types=feature_types,
            max_bin=max_bin,
            ref=ref,
            enable_categorical=enable_categorical,
            max_quantile_batches=max_quantile_batches,
        )
        return Xy

    unzipped_dict = _get_worker_parts(parts)
    it = DaskPartitionIter(
        **unzipped_dict,
        feature_types=feature_types,
        feature_names=feature_names,
        feature_weights=feature_weights,
    )
    Xy = QuantileDMatrix(
        it,
        missing=missing,
        nthread=nthread,
        max_bin=max_bin,
        ref=ref,
        enable_categorical=enable_categorical,
        max_quantile_batches=max_quantile_batches,
    )
    return Xy


def _create_dmatrix(  # pylint: disable=too-many-locals
    *,
    feature_names: Optional[FeatureNames],
    feature_types: Optional[Union[Any, List[Any]]],
    feature_weights: Optional[Any],
    missing: float,
    nthread: int,
    enable_categorical: bool,
    parts: Optional[_DataParts],
) -> DMatrix:
    """Get data that local to worker from DaskDMatrix.

    Returns
    -------
    A DMatrix object.

    """
    worker = distributed.get_worker()
    list_of_parts = parts
    if list_of_parts is None:
        msg = f"Worker {worker.address} has an empty DMatrix."
        LOGGER.warning(msg)
        Xy = DMatrix(
            np.empty((0, 0)),
            feature_names=feature_names,
            feature_types=feature_types,
            enable_categorical=enable_categorical,
        )
        return Xy

    T = TypeVar("T")

    def concat_or_none(data: Sequence[Optional[T]]) -> Optional[T]:
        if any(part is None for part in data):
            return None
        return dconcat(data)

    unzipped_dict = _get_worker_parts(list_of_parts)
    concated_dict: Dict[str, Any] = {}
    for key, value in unzipped_dict.items():
        v = concat_or_none(value)
        concated_dict[key] = v

    Xy = DMatrix(
        **concated_dict,
        missing=missing,
        feature_names=feature_names,
        feature_types=feature_types,
        nthread=nthread,
        enable_categorical=enable_categorical,
        feature_weights=feature_weights,
    )
    return Xy
