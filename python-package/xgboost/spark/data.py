"""Utilities for processing spark partitions."""
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost.compat import concat

from xgboost import DataIter, DeviceQuantileDMatrix, DMatrix


def stack_series(series: pd.Series) -> np.ndarray:
    """Stack a series of arrays."""
    array = series.to_numpy(copy=False)
    array = np.stack(array)
    return array


# Global constant for defining column alias shared between estimator and data
# processing procedures.
Alias = namedtuple("Alias", ("data", "label", "weight", "margin", "valid", "qid"))
alias = Alias("values", "label", "weight", "baseMargin", "validationIndicator", "qid")


def concat_or_none(seq: Optional[Sequence[np.ndarray]]) -> Optional[np.ndarray]:
    """Concatenate the data if it's not None."""
    if seq:
        return concat(seq)
    return None


def cache_partitions(
    iterator: Iterator[pd.DataFrame], append: Callable[[pd.DataFrame, str, bool], None]
) -> None:
    """Extract partitions from pyspark iterator. `append` is a user defined function for
    accepting new partition."""

    def make_blob(part: pd.DataFrame, is_valid: bool) -> None:
        append(part, alias.data, is_valid)
        append(part, alias.label, is_valid)
        append(part, alias.weight, is_valid)
        append(part, alias.margin, is_valid)
        append(part, alias.qid, is_valid)

    has_validation: Optional[bool] = None

    for part in iterator:
        if has_validation is None:
            has_validation = alias.valid in part.columns
        if has_validation is True:
            assert alias.valid in part.columns

        if has_validation:
            train = part.loc[~part[alias.valid], :]
            valid = part.loc[part[alias.valid], :]
        else:
            train, valid = part, None

        make_blob(train, False)
        if valid is not None:
            make_blob(valid, True)


class PartIter(DataIter):
    """Iterator for creating Quantile DMatrix from partitions."""

    def __init__(self, data: Dict[str, List], device_id: Optional[int]) -> None:
        self._iter = 0
        self._device_id = device_id
        self._data = data

        super().__init__()

    def _fetch(self, data: Optional[Sequence[pd.DataFrame]]) -> Optional[pd.DataFrame]:
        if not data:
            return None

        if self._device_id is not None:
            import cudf  # pylint: disable=import-error
            import cupy as cp  # pylint: disable=import-error

            # We must set the device after import cudf, which will change the device id to 0
            # See https://github.com/rapidsai/cudf/issues/11386
            cp.cuda.runtime.setDevice(self._device_id)
            return cudf.DataFrame(data[self._iter])

        return data[self._iter]

    def next(self, input_data: Callable) -> int:
        if self._iter == len(self._data[alias.data]):
            return 0
        input_data(
            data=self._fetch(self._data[alias.data]),
            label=self._fetch(self._data.get(alias.label, None)),
            weight=self._fetch(self._data.get(alias.weight, None)),
            base_margin=self._fetch(self._data.get(alias.margin, None)),
            qid=self._fetch(self._data.get(alias.qid, None)),
        )
        self._iter += 1
        return 1

    def reset(self) -> None:
        self._iter = 0


def _read_csr_matrix_from_unwrapped_spark_vec(part: pd.DataFrame) -> csr_matrix:
    # variables for constructing csr_matrix
    csr_indices_list, csr_indptr_list, csr_values_list = [], [0], []

    n_features = 0

    for vec_type, vec_size_, vec_indices, vec_values in zip(
        part.featureVectorType,
        part.featureVectorSize,
        part.featureVectorIndices,
        part.featureVectorValues,
    ):
        if vec_type == 0:
            # sparse vector
            vec_size = int(vec_size_)
            csr_indices = vec_indices
            csr_values = vec_values
        else:
            # dense vector
            # Note: According to spark ML VectorUDT format,
            # when type field is 1, the size field is also empty.
            # we need to check the values field to get vector length.
            vec_size = len(vec_values)
            csr_indices = np.arange(vec_size, dtype=np.int32)
            csr_values = vec_values

        if n_features == 0:
            n_features = vec_size
        assert n_features == vec_size

        csr_indices_list.append(csr_indices)
        csr_indptr_list.append(csr_indptr_list[-1] + len(csr_indices))
        csr_values_list.append(csr_values)

    csr_indptr_arr = np.array(csr_indptr_list)
    csr_indices_arr = np.concatenate(csr_indices_list)
    csr_values_arr = np.concatenate(csr_values_list)

    return csr_matrix(
        (csr_values_arr, csr_indices_arr, csr_indptr_arr), shape=(len(part), n_features)
    )


def create_dmatrix_from_partitions(
    iterator: Iterator[pd.DataFrame],
    feature_cols: Optional[Sequence[str]],
    gpu_id: Optional[int],
    kwargs: Dict[str, Any],  # use dict to make sure this parameter is passed.
    enable_sparse_data_optim: bool,
) -> Tuple[DMatrix, Optional[DMatrix]]:
    """Create DMatrix from spark data partitions. This is not particularly efficient as
    we need to convert the pandas series format to numpy then concatenate all the data.

    Parameters
    ----------
    iterator :
        Pyspark partition iterator.
    kwargs :
        Metainfo for DMatrix.

    """
    # pylint: disable=too-many-locals, too-many-statements
    train_data: Dict[str, List[np.ndarray]] = defaultdict(list)
    valid_data: Dict[str, List[np.ndarray]] = defaultdict(list)

    n_features: int = 0

    def append_m(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        nonlocal n_features
        if name in part.columns:
            array = part[name]
            if name == alias.data:
                array = stack_series(array)
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]

            if is_valid:
                valid_data[name].append(array)
            else:
                train_data[name].append(array)

    def append_m_sparse(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        nonlocal n_features

        if name == alias.data or name in part.columns:
            if name == alias.data:
                array = _read_csr_matrix_from_unwrapped_spark_vec(part)
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]
            else:
                array = part[name]

            if is_valid:
                valid_data[name].append(array)
            else:
                train_data[name].append(array)

    def append_dqm(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        """Preprocessing for DeviceQuantileDMatrix"""
        nonlocal n_features
        if name == alias.data or name in part.columns:
            if name == alias.data:
                cname = feature_cols
            else:
                cname = name

            array = part[cname]
            if name == alias.data:
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]

            if is_valid:
                valid_data[name].append(array)
            else:
                train_data[name].append(array)

    def make(values: Dict[str, List[np.ndarray]], kwargs: Dict[str, Any]) -> DMatrix:
        data = concat_or_none(values[alias.data])
        label = concat_or_none(values.get(alias.label, None))
        weight = concat_or_none(values.get(alias.weight, None))
        margin = concat_or_none(values.get(alias.margin, None))
        qid = concat_or_none(values.get(alias.qid, None))
        return DMatrix(
            data=data, label=label, weight=weight, base_margin=margin, qid=qid, **kwargs
        )

    is_dmatrix = feature_cols is None
    if is_dmatrix:
        if enable_sparse_data_optim:
            append_fn = append_m_sparse
            assert "missing" in kwargs and kwargs["missing"] == 0.0
        else:
            append_fn = append_m
        cache_partitions(iterator, append_fn)
        dtrain = make(train_data, kwargs)
    else:
        cache_partitions(iterator, append_dqm)
        it = PartIter(train_data, gpu_id)
        dtrain = DeviceQuantileDMatrix(it, **kwargs)

    dvalid = make(valid_data, kwargs) if len(valid_data) != 0 else None

    assert dtrain.num_col() == n_features
    if dvalid is not None:
        assert dvalid.num_col() == dtrain.num_col()

    return dtrain, dvalid
