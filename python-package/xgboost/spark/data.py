# pylint: disable=protected-access
"""Utilities for processing spark partitions."""
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .._typing import ArrayLike
from ..compat import concat
from ..core import DataIter, DMatrix, QuantileDMatrix
from ..sklearn import XGBModel
from .utils import get_logger


def stack_series(series: pd.Series) -> np.ndarray:
    """Stack a series of arrays."""
    array = series.to_numpy(copy=False)
    array = np.stack(array)  # type: ignore
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

    def __init__(
        self, data: Dict[str, List], device_id: Optional[int], **kwargs: Any
    ) -> None:
        self._iter = 0
        self._device_id = device_id
        self._data = data
        self._kwargs = kwargs

        super().__init__(release_data=True)

    def _fetch(self, data: Optional[Sequence[pd.DataFrame]]) -> Optional[pd.DataFrame]:
        if not data:
            return None

        if self._device_id is not None:
            import cudf
            import cupy as cp

            # We must set the device after import cudf, which will change the device id to 0
            # See https://github.com/rapidsai/cudf/issues/11386
            cp.cuda.runtime.setDevice(self._device_id)  # pylint: disable=I1101
            return cudf.DataFrame(data[self._iter])

        return data[self._iter]

    def next(self, input_data: Callable) -> bool:
        if self._iter == len(self._data[alias.data]):
            return False
        input_data(
            data=self._fetch(self._data[alias.data]),
            label=self._fetch(self._data.get(alias.label, None)),
            weight=self._fetch(self._data.get(alias.weight, None)),
            base_margin=self._fetch(self._data.get(alias.margin, None)),
            qid=self._fetch(self._data.get(alias.qid, None)),
            **self._kwargs,
        )
        self._iter += 1
        return True

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


def make_qdm(
    data: Dict[str, List[np.ndarray]],
    dev_ordinal: Optional[int],
    meta: Dict[str, Any],
    ref: Optional[DMatrix],
    params: Dict[str, Any],
) -> DMatrix:
    """Handle empty partition for QuantileDMatrix."""
    if not data:
        return QuantileDMatrix(np.empty((0, 0)), ref=ref)
    it = PartIter(data, dev_ordinal, **meta)
    m = QuantileDMatrix(it, **params, ref=ref)
    return m


def create_dmatrix_from_partitions(  # pylint: disable=too-many-arguments
    *,
    iterator: Iterator[pd.DataFrame],
    feature_cols: Optional[Sequence[str]],
    dev_ordinal: Optional[int],
    use_qdm: bool,
    kwargs: Dict[str, Any],  # use dict to make sure this parameter is passed.
    enable_sparse_data_optim: bool,
    has_validation_col: bool,
) -> Tuple[DMatrix, Optional[DMatrix]]:
    """Create DMatrix from spark data partitions.

    Parameters
    ----------
    iterator :
        Pyspark partition iterator.
    feature_cols:
        A sequence of feature names, used only when rapids plugin is enabled.
    dev_ordinal:
        Device ordinal, used when GPU is enabled.
    use_qdm :
        Whether QuantileDMatrix should be used instead of DMatrix.
    kwargs :
        Metainfo for DMatrix.
    enable_sparse_data_optim :
        Whether sparse data should be unwrapped
    has_validation:
        Whether there's validation data.

    Returns
    -------
    Training DMatrix and an optional validation DMatrix.
    """
    # pylint: disable=too-many-locals, too-many-statements
    train_data: Dict[str, List[np.ndarray]] = defaultdict(list)
    valid_data: Dict[str, List[np.ndarray]] = defaultdict(list)

    n_features: int = 0

    def append_m(part: pd.DataFrame, name: str, is_valid: bool) -> None:
        nonlocal n_features
        if name == alias.data or name in part.columns:
            if (
                name == alias.data
                and feature_cols is not None
                and part[feature_cols].shape[0] > 0  # guard against empty partition
            ):
                array: Optional[np.ndarray] = part[feature_cols]
            elif part[name].shape[0] > 0:
                array = part[name]
                if name == alias.data:
                    # For the array/vector typed case.
                    array = stack_series(array)
            else:
                array = None

            if name == alias.data and array is not None:
                if n_features == 0:
                    n_features = array.shape[1]
                assert n_features == array.shape[1]

            if array is None:
                return

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

    def make(values: Dict[str, List[np.ndarray]], kwargs: Dict[str, Any]) -> DMatrix:
        if len(values) == 0:
            get_logger("XGBoostPySpark").warning(
                "Detected an empty partition in the training data. Consider to enable"
                " repartition_random_shuffle"
            )
            # We must construct an empty DMatrix to bypass the AllReduce
            return DMatrix(data=np.empty((0, 0)), **kwargs)

        data = concat_or_none(values[alias.data])
        label = concat_or_none(values.get(alias.label, None))
        weight = concat_or_none(values.get(alias.weight, None))
        margin = concat_or_none(values.get(alias.margin, None))
        qid = concat_or_none(values.get(alias.qid, None))
        return DMatrix(
            data=data, label=label, weight=weight, base_margin=margin, qid=qid, **kwargs
        )

    if enable_sparse_data_optim:
        append_fn = append_m_sparse
        assert "missing" in kwargs and kwargs["missing"] == 0.0
    else:
        append_fn = append_m

    def split_params() -> Tuple[Dict[str, Any], Dict[str, Union[int, float, bool]]]:
        # FIXME(jiamingy): we really need a better way to bridge distributed frameworks
        # to XGBoost native interface and prevent scattering parameters like this.

        # parameters that are not related to data.
        non_data_keys = (
            "max_bin",
            "missing",
            "silent",
            "nthread",
            "enable_categorical",
        )
        non_data_params = {}
        meta = {}
        for k, v in kwargs.items():
            if k in non_data_keys:
                non_data_params[k] = v
            else:
                meta[k] = v
        return meta, non_data_params

    meta, params = split_params()

    if feature_cols is not None and use_qdm:
        cache_partitions(iterator, append_fn)
        dtrain: DMatrix = make_qdm(train_data, dev_ordinal, meta, None, params)
    elif feature_cols is not None and not use_qdm:
        cache_partitions(iterator, append_fn)
        dtrain = make(train_data, kwargs)
    elif feature_cols is None and use_qdm:
        cache_partitions(iterator, append_fn)
        dtrain = make_qdm(train_data, dev_ordinal, meta, None, params)
    else:
        cache_partitions(iterator, append_fn)
        dtrain = make(train_data, kwargs)

    # Using has_validation_col here to indicate if there is validation col
    # instead of getting it from iterator, since the iterator may be empty
    # in some special case. That is to say, we must ensure every worker
    # construct DMatrix even there is no data since we need to ensure every
    # worker do the AllReduce when constructing DMatrix, or else it may hang
    # forever.
    if has_validation_col:
        if use_qdm:
            dvalid: Optional[DMatrix] = make_qdm(
                valid_data, dev_ordinal, meta, dtrain, params
            )
        else:
            dvalid = make(valid_data, kwargs) if has_validation_col else None
    else:
        dvalid = None

    if dvalid is not None:
        assert dvalid.num_col() == dtrain.num_col()

    return dtrain, dvalid


def pred_contribs(
    model: XGBModel,
    data: ArrayLike,
    base_margin: Optional[ArrayLike] = None,
    strict_shape: bool = False,
) -> np.ndarray:
    """Predict contributions with data with the full model."""
    iteration_range = model._get_iteration_range(None)
    data_dmatrix = DMatrix(
        data,
        base_margin=base_margin,
        missing=model.missing,
        nthread=model.n_jobs,
        feature_types=model.feature_types,
        feature_weights=model.feature_weights,
        enable_categorical=model.enable_categorical,
    )
    return model.get_booster().predict(
        data_dmatrix,
        pred_contribs=True,
        validate_features=False,
        iteration_range=iteration_range,
        strict_shape=strict_shape,
    )
