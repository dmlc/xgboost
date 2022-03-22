# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import pytest
import sys

sys.path.append("tests/python")
import testing as tm


class TestDeviceQuantileDMatrix:
    def test_dmatrix_numpy_init(self):
        data = np.random.randn(5, 5)
        with pytest.raises(TypeError, match='is not supported'):
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

    @pytest.mark.skipif(**tm.no_cupy())
    def test_metainfo(self) -> None:
        import cupy as cp
        rng = cp.random.RandomState(1994)

        rows = 10
        cols = 3
        data = rng.randn(rows, cols)

        labels = rng.randn(rows)

        fw = rng.randn(rows)
        fw -= fw.min()

        m = xgb.DeviceQuantileDMatrix(data=data, label=labels, feature_weights=fw)

        got_fw = m.get_float_info("feature_weights")
        got_labels = m.get_label()

        cp.testing.assert_allclose(fw, got_fw)
        cp.testing.assert_allclose(labels, got_labels)
