import cupy as cp
import numpy as np
import pytest

import xgboost as xgb


# Test for integer overflow or out of memory exceptions
def test_large_input():
    available_bytes, _ = cp.cuda.runtime.memGetInfo()
    # 15 GB
    required_bytes = 1.5e+10
    if available_bytes < required_bytes:
        pytest.skip("Not enough memory on this device")
    n = 1000
    m = ((1 << 31) + n - 1) // n
    assert (np.log2(m * n) > 31)
    X = cp.ones((m, n), dtype=np.float32)
    y = cp.ones(m)
    dmat = xgb.QuantileDMatrix(X, y)
    booster = xgb.train({"tree_method": "gpu_hist", "max_depth": 1}, dmat, 1)
    del y
    booster.inplace_predict(X)
