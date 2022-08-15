import numpy as np
import xgboost as xgb
import pytest
import sys

sys.path.append("tests/python")
import testing as tm
import test_quantile_dmatrix as tqd


class TestDeviceQuantileDMatrix:
    cputest = tqd.TestQuantileDMatrix()

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dmatrix_feature_weights(self) -> None:
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
    def test_dmatrix_cupy_init(self) -> None:
        import cupy as cp
        data = cp.random.randn(5, 5)
        xgb.DeviceQuantileDMatrix(data, cp.ones(5, dtype=np.float64))

    @pytest.mark.skipif(**tm.no_cupy())
    def test_from_host(self) -> None:
        import cupy as cp
        n_samples = 64
        n_features = 3
        X, y, w = tm.make_batches(
            n_samples, n_features=n_features, n_batches=1, use_cupy=False
        )
        Xy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0])
        booster_0 = xgb.train({"tree_method": "gpu_hist"}, Xy, num_boost_round=4)

        X[0] = cp.array(X[0])
        y[0] = cp.array(y[0])
        w[0] = cp.array(w[0])

        Xy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0])
        booster_1 = xgb.train({"tree_method": "gpu_hist"}, Xy, num_boost_round=4)
        cp.testing.assert_allclose(
            booster_0.inplace_predict(X[0]), booster_1.inplace_predict(X[0])
        )

        with pytest.raises(ValueError, match="not initialized with CPU"):
            # Training on CPU with GPU data is not supported.
            xgb.train({"tree_method": "hist"}, Xy, num_boost_round=4)

        with pytest.raises(ValueError, match=r"Only.*hist.*"):
            xgb.train({"tree_method": "approx"}, Xy, num_boost_round=4)

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

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.skipif(**tm.no_cudf())
    def test_ref_dmatrix(self) -> None:
        import cupy as cp
        rng = cp.random.RandomState(1994)
        self.cputest.run_ref_dmatrix(rng, "gpu_hist", False)
