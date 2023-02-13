import sys

import numpy as np
import pytest
from hypothesis import given, settings, strategies

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.data import check_inf

sys.path.append("tests/python")
import test_quantile_dmatrix as tqd


class TestQuantileDMatrix:
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
            cp.array(m.get_float_info("feature_weights")),
            feature_weights.astype(np.float32),
        )

    @pytest.mark.skipif(**tm.no_cupy())
    def test_dmatrix_cupy_init(self) -> None:
        import cupy as cp

        data = cp.random.randn(5, 5)
        xgb.QuantileDMatrix(data, cp.ones(5, dtype=np.float64))

    @pytest.mark.parametrize(
        "on_device,tree_method",
        [(True, "hist"), (False, "gpu_hist"), (False, "hist"), (True, "gpu_hist")],
    )
    def test_initialization(self, on_device: bool, tree_method: str) -> None:
        n_samples, n_features, max_bin = 64, 3, 16
        X, y, w = tm.make_batches(
            n_samples,
            n_features=n_features,
            n_batches=1,
            use_cupy=on_device,
        )

        # Init SparsePage
        Xy = xgb.DMatrix(X[0], y[0], weight=w[0])
        # Init GIDX/Ellpack
        xgb.train(
            {"tree_method": tree_method, "max_bin": max_bin},
            Xy,
            num_boost_round=1,
        )
        # query cuts from GIDX/Ellpack
        qXy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin, ref=Xy)
        tm.predictor_equal(Xy, qXy)
        with pytest.raises(ValueError, match="Inconsistent"):
            # max_bin changed.
            xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin - 1, ref=Xy)

        # No error, DMatrix can be modified for different training session.
        xgb.train(
            {"tree_method": tree_method, "max_bin": max_bin - 1},
            Xy,
            num_boost_round=1,
        )

        # Init Ellpack/GIDX
        Xy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin)
        # Init GIDX/Ellpack
        xgb.train(
            {"tree_method": tree_method, "max_bin": max_bin},
            Xy,
            num_boost_round=1,
        )
        # query cuts from GIDX/Ellpack
        qXy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin, ref=Xy)
        tm.predictor_equal(Xy, qXy)
        with pytest.raises(ValueError, match="Inconsistent"):
            # max_bin changed.
            xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin - 1, ref=Xy)

        Xy = xgb.DMatrix(X[0], y[0], weight=w[0])
        booster0 = xgb.train(
            {"tree_method": "hist", "max_bin": max_bin, "max_depth": 4},
            Xy,
            num_boost_round=1,
        )
        booster1 = xgb.train(
            {"tree_method": "gpu_hist", "max_bin": max_bin, "max_depth": 4},
            Xy,
            num_boost_round=1,
        )
        qXy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin, ref=Xy)
        predt0 = booster0.predict(qXy)
        predt1 = booster1.predict(qXy)
        np.testing.assert_allclose(predt0, predt1)

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.parametrize(
        "tree_method,max_bin",
        [("hist", 16), ("gpu_hist", 16), ("hist", 64), ("gpu_hist", 64)],
    )
    def test_interoperability(self, tree_method: str, max_bin: int) -> None:
        import cupy as cp

        n_samples = 64
        n_features = 3
        X, y, w = tm.make_batches(
            n_samples, n_features=n_features, n_batches=1, use_cupy=False
        )
        # from CPU
        Xy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin)
        booster_0 = xgb.train(
            {"tree_method": tree_method, "max_bin": max_bin}, Xy, num_boost_round=4
        )

        X[0] = cp.array(X[0])
        y[0] = cp.array(y[0])
        w[0] = cp.array(w[0])

        # from GPU
        Xy = xgb.QuantileDMatrix(X[0], y[0], weight=w[0], max_bin=max_bin)
        booster_1 = xgb.train(
            {"tree_method": tree_method, "max_bin": max_bin}, Xy, num_boost_round=4
        )
        cp.testing.assert_allclose(
            booster_0.inplace_predict(X[0]), booster_1.inplace_predict(X[0])
        )

        with pytest.raises(ValueError, match=r"Only.*hist.*"):
            xgb.train(
                {"tree_method": "approx", "max_bin": max_bin}, Xy, num_boost_round=4
            )

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

        m = xgb.QuantileDMatrix(data=data, label=labels, feature_weights=fw)

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

    @given(
        strategies.integers(1, 1000),
        strategies.integers(1, 100),
        strategies.fractions(0, 0.99),
    )
    @settings(print_blob=True, deadline=None)
    def test_to_csr(self, n_samples, n_features, sparsity) -> None:
        import cupy as cp

        X, y = tm.make_sparse_regression(n_samples, n_features, sparsity, False)
        h_X = X.astype(np.float32)

        csr = h_X
        h_X = X.toarray()
        h_X[h_X == 0] = np.nan

        h_m = xgb.QuantileDMatrix(data=h_X)
        h_ret = h_m.get_data()

        d_X = cp.array(h_X)

        d_m = xgb.QuantileDMatrix(data=d_X, label=y)
        d_ret = d_m.get_data()

        np.testing.assert_equal(csr.indptr, d_ret.indptr)
        np.testing.assert_equal(csr.indices, d_ret.indices)

        np.testing.assert_equal(h_ret.indptr, d_ret.indptr)
        np.testing.assert_equal(h_ret.indices, d_ret.indices)

        booster = xgb.train(
            {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}, dtrain=d_m
        )

        np.testing.assert_allclose(
            booster.predict(d_m),
            booster.predict(xgb.DMatrix(d_m.get_data())),
            atol=1e-6,
        )

    def test_ltr(self) -> None:
        import cupy as cp
        X, y, qid, w = tm.make_ltr(100, 3, 3, 5)
        # make sure GPU is used to run sketching.
        cpX = cp.array(X)
        Xy_qdm = xgb.QuantileDMatrix(cpX, y, qid=qid, weight=w)
        Xy = xgb.DMatrix(X, y, qid=qid, weight=w)
        xgb.train({"tree_method": "gpu_hist", "objective": "rank:ndcg"}, Xy)

        from_dm = xgb.QuantileDMatrix(X, weight=w, ref=Xy)
        from_qdm = xgb.QuantileDMatrix(X, weight=w, ref=Xy_qdm)

        assert tm.predictor_equal(from_qdm, from_dm)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_check_inf(self) -> None:
        import cupy as cp

        rng = cp.random.default_rng(1994)
        check_inf(rng)
