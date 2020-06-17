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
        with pytest.raises(AssertionError, match='is not supported for DeviceQuantileDMatrix'):
            dm = xgb.DeviceQuantileDMatrix(data, np.ones(5, dtype=np.float64))

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dmatrix_cupy_init(self):
        import cupy as cp
        data = cp.random.randn(5, 5)
        dm = xgb.DeviceQuantileDMatrix(data, cp.ones(5, dtype=np.float64))
