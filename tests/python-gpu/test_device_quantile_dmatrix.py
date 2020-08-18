# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import unittest
import pytest
import sys

sys.path.append("tests/python")
import testing as tm


class TestDeviceQuantileDMatrix(unittest.TestCase):
    def test_dmatrix_numpy_init(self):
        data = np.random.randn(5, 5)
        with pytest.raises(TypeError,
                           match='is not supported for DeviceQuantileDMatrix'):
            xgb.DeviceQuantileDMatrix(data, np.ones(5, dtype=np.float64))

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dmatrix_feature_weights(self):
        import cupy as cp
        rng = cp.random.RandomState(1994)
        data = rng.randn(5, 5)
        m = xgb.DMatrix(data)

        feature_weights = rng.uniform(size=5)
        m.set_info(feature_weights=feature_weights)

        cp.testing.assert_array_equal(
            cp.array(m.get_float_info('feature_weights')),
            feature_weights.astype(np.float32))

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dmatrix_cupy_init(self):
        import cupy as cp
        data = cp.random.randn(5, 5)
        xgb.DeviceQuantileDMatrix(data, cp.ones(5, dtype=np.float64))
