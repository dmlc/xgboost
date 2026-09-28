import cupy as cp
import numpy as np
import pytest

import xgboost as xgb


# Test for integer overflow or out of memory exceptions
def test_large_input() -> None:
    available_bytes, _ = cp.cuda.runtime.memGetInfo()
    # 15 GB
    required_bytes = 1.5e10
    if available_bytes < required_bytes:
        pytest.skip("Not enough memory on this device")
    n = 1000
    m = ((1 << 31) + n - 1) // n
    assert np.log2(m * n) > 31
    X = cp.ones((m, n), dtype=np.float32)
    y = cp.ones(m)
    w = cp.ones(m)
    dmat = xgb.QuantileDMatrix(X, y, weight=w)
    booster = xgb.train(
        {"tree_method": "hist", "max_depth": 1, "device": "cuda"}, dmat, 1
    )
    del y
    booster.inplace_predict(X)


def test_inplace_predict_row_offset_beyond_uint32() -> None:
    n_features = 96
    n_rows = (1 << 32) // n_features + 2
    available_bytes, _ = cp.cuda.runtime.memGetInfo()
    if available_bytes < 5e9:
        pytest.skip("Not enough memory on this device")
    if cp.cuda.Device().attributes["MaxSharedMemoryPerBlock"] < 4 * 128 * n_features:
        pytest.skip("requires the shared-memory prediction path")

    train = np.zeros((8, n_features), dtype=np.float32)
    train[4:, 0] = 1
    booster = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 1, "min_child_weight": 0},
        xgb.DMatrix(train, label=train[:, 0]),
        num_boost_round=4,
    )
    booster.set_param({"device": "cuda"})

    data = cp.zeros((n_rows, n_features), dtype=cp.uint8)
    data[-1, 0] = 1
    expected = booster.inplace_predict(data[-1:])
    baseline = booster.inplace_predict(data[:1])
    assert cp.asnumpy(expected)[0] != cp.asnumpy(baseline)[0]

    actual = booster.inplace_predict(data)
    np.testing.assert_allclose(cp.asnumpy(actual[-1:]), cp.asnumpy(expected))
