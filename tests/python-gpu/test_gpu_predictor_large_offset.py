"""Opt-in regression for GPU in-place prediction beyond 2**32 input cells."""

import os

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.skipif(
    os.environ.get("XGBOOST_TEST_LARGE_GPU") != "1",
    reason="requires at least 4.5 GiB of free GPU memory",
)
def test_inplace_predict_row_offset_beyond_uint32() -> None:
    cp = pytest.importorskip("cupy")
    n_features = 96
    n_rows = (1 << 32) // n_features + 2
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
