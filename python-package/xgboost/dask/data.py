# pylint: disable=too-many-arguments
"""Copyright 2019-2024, XGBoost contributors"""

import logging
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import distributed
import numpy as np
from dask import dataframe as dd

from .._typing import _T, FeatureNames
from ..compat import concat
from ..core import DataIter, DMatrix, QuantileDMatrix

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
