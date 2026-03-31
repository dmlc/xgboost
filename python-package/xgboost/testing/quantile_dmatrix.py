"""QuantileDMatrix related tests."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

import xgboost as xgb

from .data import make_batches, make_categorical

MAX_NORMALIZED_RANK_ERROR = 2.0
MAX_WEIGHTED_NORMALIZED_RANK_ERROR = 14.0


@dataclass(frozen=True)
class _RankContext:
    sorted_x: np.ndarray
    prefix_sum: np.ndarray
    total_weight: float
    num_cuts: int
    avg_bin_weight: float


def _to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "get"):
        data = data.get()
    elif hasattr(data, "to_pandas"):
        data = data.to_pandas()
    if hasattr(data, "to_numpy"):
        data = data.to_numpy()
    return np.asarray(data)


def _distance_to_interval(target: float, lo: float, hi: float) -> float:
    if target < lo:
        return lo - target
    if target > hi:
        return target - hi
    return 0.0


def _prepare_validation_input(
    x: Any, w: Optional[Any]
) -> tuple[Any, np.ndarray, float]:
    x_data = x.get() if hasattr(x, "get") else x
    if hasattr(x_data, "to_pandas"):
        x_data = x_data.to_pandas()

    if w is None:
        weights = np.ones(x_data.shape[0], dtype=np.float64)
    else:
        weights = _to_numpy(w).astype(np.float64, copy=False)
        assert weights.ndim == 1
        assert weights.shape[0] == x_data.shape[0]

    max_rank_error = (
        MAX_NORMALIZED_RANK_ERROR
        if np.all(weights == 1.0)
        else MAX_WEIGHTED_NORMALIZED_RANK_ERROR
    )
    return x_data, weights, max_rank_error


def _column_getter(
    x_data: Any, weights: np.ndarray
) -> tuple[int, Callable[[int], tuple[np.ndarray, np.ndarray]]]:
    if hasattr(x_data, "tocsc") and hasattr(x_data, "indptr"):
        csc = x_data.tocsc()

        def get_sparse_column(fidx: int) -> tuple[np.ndarray, np.ndarray]:
            beg = int(csc.indptr[fidx])
            end = int(csc.indptr[fidx + 1])
            indices = csc.indices[beg:end]
            column = np.asarray(csc.data[beg:end])
            return column, weights[indices]

        return csc.shape[1], get_sparse_column

    x_dense = _to_numpy(x_data)
    assert x_dense.ndim == 2

    def get_dense_column(fidx: int) -> tuple[np.ndarray, np.ndarray]:
        column = x_dense[:, fidx]
        valid = ~np.isnan(column)
        return column[valid], weights[valid]

    return x_dense.shape[1], get_dense_column


def _sorted_rank_state(
    column: np.ndarray, column_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    sorted_idx = np.argsort(column, kind="stable")
    sorted_x = column[sorted_idx]
    sorted_w = column_w[sorted_idx]
    prefix_sum = np.concatenate(([0.0], np.cumsum(sorted_w, dtype=np.float64)))
    return sorted_x, prefix_sum, float(prefix_sum[-1])


def _make_rank_context(
    column: np.ndarray, column_w: np.ndarray, column_cuts: np.ndarray
) -> _RankContext | None:
    sorted_x, prefix_sum, total_weight = _sorted_rank_state(column, column_w)
    if total_weight == 0.0:
        return None
    return _RankContext(
        sorted_x=sorted_x,
        prefix_sum=prefix_sum,
        total_weight=total_weight,
        num_cuts=column_cuts.shape[0],
        avg_bin_weight=total_weight / float(column_cuts.shape[0]),
    )


def _rank_error_candidate(
    cut_idx: int,
    cut: float,
    rank_ctx: _RankContext,
) -> tuple[float, dict[str, float | int]]:
    rank_lo = float(
        rank_ctx.prefix_sum[np.searchsorted(rank_ctx.sorted_x, cut, side="left")]
    )
    rank_hi = float(
        rank_ctx.prefix_sum[np.searchsorted(rank_ctx.sorted_x, cut, side="right")]
    )
    target_rank = ((cut_idx + 1) * rank_ctx.total_weight) / float(rank_ctx.num_cuts)
    absolute_error = _distance_to_interval(target_rank, rank_lo, rank_hi)
    return absolute_error / rank_ctx.avg_bin_weight, {
        "cut": cut_idx,
        "absolute_error": absolute_error,
        "target_rank": target_rank,
        "rank_lo": rank_lo,
        "rank_hi": rank_hi,
    }


def _max_rank_error_for_column(
    column: np.ndarray, column_w: np.ndarray, column_cuts: np.ndarray
) -> tuple[float, str]:
    rank_ctx = _make_rank_context(column, column_w, column_cuts)
    if rank_ctx is None:
        return 0.0, ""

    max_error = 0.0
    max_state = {
        "cut": 0,
        "absolute_error": 0.0,
        "target_rank": 0.0,
        "rank_lo": 0.0,
        "rank_hi": 0.0,
    }
    for cut_idx, cut in enumerate(column_cuts[:-1]):
        error, state = _rank_error_candidate(cut_idx, cut, rank_ctx)
        if error > max_error:
            max_error = error
            max_state = state

    details = (
        f"cut={max_state['cut']}, normalized_error={max_error}, "
        f"absolute_error={max_state['absolute_error']}, "
        f"target_rank={max_state['target_rank']}, rank_lo={max_state['rank_lo']}, "
        f"rank_hi={max_state['rank_hi']}, total_weight={rank_ctx.total_weight}, "
        f"num_cuts={column_cuts.shape[0]}"
    )
    return max_error, details


def _assert_feature_rank_error(
    indptr: np.ndarray,
    cuts: np.ndarray,
    get_column: Callable[[int], tuple[np.ndarray, np.ndarray]],
    fidx: int,
    max_normalized_rank_error: float,
) -> None:
    column, column_w = get_column(fidx)
    if column.shape[0] == 0:
        return

    beg = int(indptr[fidx])
    end = int(indptr[fidx + 1])
    column_cuts = cuts[beg:end]
    assert np.all(np.diff(column_cuts) >= 0.0)
    if column_cuts.shape[0] <= 1:
        return

    max_error, details = _max_rank_error_for_column(column, column_w, column_cuts)
    assert max_error <= max_normalized_rank_error, f"feature={fidx}, {details}"


def assert_cut_rank_error_within_tolerance(
    indptr: np.ndarray,
    cuts: np.ndarray,
    x: Any,
    w: Optional[Any] = None,
    max_normalized_rank_error: Optional[float] = None,
) -> None:
    """Assert that every numerical feature cut stays within the allowed rank error."""
    x_data, weights, default_rank_error = _prepare_validation_input(x, w)
    if max_normalized_rank_error is None:
        max_normalized_rank_error = default_rank_error

    n_features, get_column = _column_getter(x_data, weights)
    for fidx in range(n_features):
        _assert_feature_rank_error(
            indptr, cuts, get_column, fidx, max_normalized_rank_error
        )


def check_ref_quantile_cut(device: str) -> None:
    """Check obtaining the same cut values given a reference."""
    X, y, _ = (
        data[0]
        for data in make_batches(
            n_samples_per_batch=8192,
            n_features=16,
            n_batches=1,
            use_cupy=device.startswith("cuda"),
        )
    )

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    Xy_train = xgb.QuantileDMatrix(X_train, y_train)
    Xy_valid = xgb.QuantileDMatrix(X_valid, y_valid, ref=Xy_train)

    cut_train = Xy_train.get_quantile_cut()
    cut_valid = Xy_valid.get_quantile_cut()

    np.testing.assert_allclose(cut_train[0], cut_valid[0])
    np.testing.assert_allclose(cut_train[1], cut_valid[1])
    assert_cut_rank_error_within_tolerance(cut_train[0], cut_train[1], X_train)

    Xy_valid = xgb.QuantileDMatrix(X_valid, y_valid)
    cut_valid = Xy_valid.get_quantile_cut()
    assert not np.allclose(cut_train[1], cut_valid[1])
    assert_cut_rank_error_within_tolerance(cut_valid[0], cut_valid[1], X_valid)


def check_categorical_strings(device: str) -> None:
    """Check string inputs."""
    if device == "cpu":
        pd = pytest.importorskip("pandas")
    else:
        pd = pytest.importorskip("cudf")

    n_categories = 32
    X, y = make_categorical(
        1024,
        8,
        n_categories,
        onehot=False,
        cat_dtype=np.str_,
        cat_ratio=0.5,
        shuffle=True,
    )
    X = pd.DataFrame(X)

    Xy = xgb.QuantileDMatrix(X, y, enable_categorical=True)
    assert Xy.num_col() == 8
    cuts = Xy.get_quantile_cut()
    indptr = cuts[0]
    values = cuts[1]
    for i in range(1, len(indptr)):
        f_idx = i - 1
        if isinstance(X[X.columns[f_idx]].dtype, pd.CategoricalDtype):
            beg, end = indptr[f_idx], indptr[i]
            col = values[beg:end]
            np.testing.assert_allclose(col, np.arange(0, n_categories))
