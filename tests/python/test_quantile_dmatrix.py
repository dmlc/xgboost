from typing import Any, Dict, List

import numpy as np
import pytest
from hypothesis import given, settings, strategies
from scipy import sparse

import xgboost as xgb
from xgboost.testing import (
    IteratorForTest,
    make_batches,
    make_batches_sparse,
    make_categorical,
    make_ltr,
    make_sparse_regression,
    predictor_equal,
)
from xgboost.testing.data import check_inf, np_dtypes
from xgboost.testing.data_iter import run_mixed_sparsity
from xgboost.testing.quantile_dmatrix import (
    check_categorical_strings,
    check_ref_quantile_cut,
)


class TestQuantileDMatrix:
    def test_basic(self) -> None:
        """Checks for np array, list, tuple."""
        n_samples = 234
        n_features = 8

        rng = np.random.default_rng()
        X = rng.normal(loc=0, scale=3, size=n_samples * n_features).reshape(
            n_samples, n_features
        )
        y = rng.normal(0, 3, size=n_samples)
        Xy = xgb.QuantileDMatrix(X, y)
        assert Xy.num_row() == n_samples
        assert Xy.num_col() == n_features

        X = sparse.random(n_samples, n_features, density=0.1, format="csr")
        Xy = xgb.QuantileDMatrix(X, y)
        assert Xy.num_row() == n_samples
        assert Xy.num_col() == n_features

        X = sparse.random(n_samples, n_features, density=0.8, format="csr")
        Xy = xgb.QuantileDMatrix(X, y)
        assert Xy.num_row() == n_samples
        assert Xy.num_col() == n_features

        n_samples = 64
        data = []
        for f in range(n_samples):
            row = [f] * n_features
            data.append(row)
        assert np.array(data).shape == (n_samples, n_features)
        Xy = xgb.QuantileDMatrix(data, max_bin=256)
        assert Xy.num_row() == n_samples
        assert Xy.num_col() == n_features
        r = np.arange(1.0, n_samples)
        np.testing.assert_allclose(Xy.get_data().toarray()[1:, 0], r)

    def test_categorical_strings(self) -> None:
        check_categorical_strings("cpu")

    def test_error(self):
        from sklearn.model_selection import train_test_split

        rng = np.random.default_rng(1994)
        X, y = make_categorical(
            n_samples=128, n_features=2, n_categories=3, onehot=False
        )
        reg = xgb.XGBRegressor(tree_method="hist", enable_categorical=True)
        w = rng.uniform(0, 1, size=y.shape[0])

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, w, random_state=1994
        )

        with pytest.raises(ValueError, match="sample weight"):
            reg.fit(
                X,
                y,
                sample_weight=w_train,
                eval_set=[(X_test, y_test)],
                sample_weight_eval_set=[w_test],
            )

        with pytest.raises(ValueError, match="sample weight"):
            reg.fit(
                X_train,
                y_train,
                sample_weight=w,
                eval_set=[(X_test, y_test)],
                sample_weight_eval_set=[w_test],
            )

    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.8, 0.9])
    def test_with_iterator(self, sparsity: float) -> None:
        n_samples_per_batch = 317
        n_features = 8
        n_batches = 7

        if sparsity == 0.0:
            it = IteratorForTest(
                *make_batches(n_samples_per_batch, n_features, n_batches, False),
                cache=None,
            )
        else:
            it = IteratorForTest(
                *make_batches_sparse(
                    n_samples_per_batch, n_features, n_batches, sparsity
                ),
                cache=None,
            )
        Xy = xgb.QuantileDMatrix(it)
        assert Xy.num_row() == n_samples_per_batch * n_batches
        assert Xy.num_col() == n_features

    def test_different_size(self) -> None:
        n_samples_per_batch = 317
        n_features = 8
        n_batches = 7

        it = IteratorForTest(
            *make_batches(
                n_samples_per_batch, n_features, n_batches, False, vary_size=True
            ),
            cache=None,
        )
        Xy = xgb.QuantileDMatrix(it)
        assert Xy.num_row() == 2429
        X, y, w = it.as_arrays()
        Xy1 = xgb.QuantileDMatrix(X, y, weight=w)
        assert predictor_equal(Xy, Xy1)

    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.5, 0.8, 0.9])
    def test_training(self, sparsity: float) -> None:
        n_samples_per_batch = 317
        n_features = 8
        n_batches = 7
        if sparsity == 0.0:
            it = IteratorForTest(
                *make_batches(n_samples_per_batch, n_features, n_batches, False),
                cache=None,
            )
        else:
            it = IteratorForTest(
                *make_batches_sparse(
                    n_samples_per_batch, n_features, n_batches, sparsity
                ),
                cache=None,
            )

        parameters = {"tree_method": "hist", "max_bin": 256}
        Xy_it = xgb.QuantileDMatrix(it, max_bin=parameters["max_bin"])
        from_it = xgb.train(parameters, Xy_it)

        X, y, w = it.as_arrays()
        w_it = Xy_it.get_weight()
        np.testing.assert_allclose(w_it, w)

        Xy_arr = xgb.DMatrix(X, y, weight=w)
        from_arr = xgb.train(parameters, Xy_arr)

        np.testing.assert_allclose(from_arr.predict(Xy_it), from_it.predict(Xy_arr))

        y -= y.min()
        y += 0.01
        Xy = xgb.QuantileDMatrix(X, y, weight=w)
        with pytest.raises(ValueError, match=r"Only.*hist.*"):
            parameters = {
                "tree_method": "approx",
                "max_bin": 256,
                "objective": "reg:gamma",
            }
            xgb.train(parameters, Xy)

    def run_ref_dmatrix(self, rng: Any, device: str, enable_cat: bool) -> None:
        n_samples, n_features = 2048, 17
        if enable_cat:
            X, y = make_categorical(
                n_samples, n_features, n_categories=13, onehot=False
            )
            if device == "cuda":
                import cudf

                X = cudf.from_pandas(X)
                y = cudf.from_pandas(y)
        else:
            X = rng.normal(loc=0, scale=3, size=n_samples * n_features).reshape(
                n_samples, n_features
            )
            y = rng.normal(0, 3, size=n_samples)

        # Use ref
        Xy = xgb.QuantileDMatrix(X, y, enable_categorical=enable_cat)
        Xy_valid: xgb.DMatrix = xgb.QuantileDMatrix(
            X, y, ref=Xy, enable_categorical=enable_cat
        )
        qdm_results: Dict[str, Dict[str, List[float]]] = {}
        xgb.train(
            {"tree_method": "hist", "device": device},
            Xy,
            evals=[(Xy, "Train"), (Xy_valid, "valid")],
            evals_result=qdm_results,
        )
        np.testing.assert_allclose(
            qdm_results["Train"]["rmse"], qdm_results["valid"]["rmse"]
        )
        # No ref
        Xy_valid = xgb.DMatrix(X, y, enable_categorical=enable_cat)
        qdm_results = {}
        xgb.train(
            {"tree_method": "hist", "device": device},
            Xy,
            evals=[(Xy, "Train"), (Xy_valid, "valid")],
            evals_result=qdm_results,
        )
        np.testing.assert_allclose(
            qdm_results["Train"]["rmse"], qdm_results["valid"]["rmse"]
        )

        # Different number of features
        Xy = xgb.QuantileDMatrix(X, y, enable_categorical=enable_cat)
        dXy = xgb.DMatrix(X, y, enable_categorical=enable_cat)

        n_samples, n_features = 256, 15
        X = rng.normal(loc=0, scale=3, size=n_samples * n_features).reshape(
            n_samples, n_features
        )
        y = rng.normal(0, 3, size=n_samples)
        with pytest.raises(ValueError, match=r".*features\."):
            xgb.QuantileDMatrix(X, y, ref=Xy, enable_categorical=enable_cat)

        # Compare training results
        n_samples, n_features = 256, 17
        if enable_cat:
            X, y = make_categorical(n_samples, n_features, 13, onehot=False)
            if device == "cuda":
                import cudf

                X = cudf.from_pandas(X)
                y = cudf.from_pandas(y)
        else:
            X = rng.normal(loc=0, scale=3, size=n_samples * n_features).reshape(
                n_samples, n_features
            )
            y = rng.normal(0, 3, size=n_samples)
        Xy_valid = xgb.QuantileDMatrix(X, y, ref=Xy, enable_categorical=enable_cat)
        # use DMatrix as ref
        Xy_valid_d = xgb.QuantileDMatrix(X, y, ref=dXy, enable_categorical=enable_cat)
        dXy_valid = xgb.DMatrix(X, y, enable_categorical=enable_cat)

        qdm_results = {}
        xgb.train(
            {"tree_method": "hist", "device": device},
            Xy,
            evals=[(Xy, "Train"), (Xy_valid, "valid")],
            evals_result=qdm_results,
        )

        dm_results: Dict[str, Dict[str, List[float]]] = {}
        xgb.train(
            {"tree_method": "hist", "device": device},
            dXy,
            evals=[(dXy, "Train"), (dXy_valid, "valid"), (Xy_valid_d, "dvalid")],
            evals_result=dm_results,
        )
        np.testing.assert_allclose(
            dm_results["Train"]["rmse"], qdm_results["Train"]["rmse"]
        )
        np.testing.assert_allclose(
            dm_results["valid"]["rmse"], qdm_results["valid"]["rmse"]
        )
        np.testing.assert_allclose(
            dm_results["dvalid"]["rmse"], qdm_results["valid"]["rmse"]
        )

        Xy_valid = xgb.QuantileDMatrix(X, y, enable_categorical=enable_cat)
        with pytest.raises(ValueError, match="should be used as a reference"):
            xgb.train(
                {"device": device}, dXy, evals=[(dXy, "Train"), (Xy_valid, "Valid")]
            )

    def test_ref_quantile_cut(self) -> None:
        check_ref_quantile_cut("cpu")

    @pytest.mark.parametrize("enable_cat", [True, False])
    def test_ref_dmatrix(self, enable_cat: bool) -> None:
        rng = np.random.RandomState(1994)
        self.run_ref_dmatrix(rng, "cpu", enable_cat)

    @pytest.mark.parametrize("sparsity", [0.0, 0.5])
    def test_predict(self, sparsity: float) -> None:
        n_samples, n_features = 256, 4
        X, y = make_categorical(
            n_samples, n_features, n_categories=13, onehot=False, sparsity=sparsity
        )
        Xy = xgb.DMatrix(X, y, enable_categorical=True)

        booster = xgb.train({"tree_method": "hist"}, Xy)

        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        a = booster.predict(Xy)
        qXy = xgb.QuantileDMatrix(X, y, enable_categorical=True)
        b = booster.predict(qXy)
        np.testing.assert_allclose(a, b)

    def test_ltr(self) -> None:
        X, y, qid, w = make_ltr(100, 3, 3, 5)
        Xy_qdm = xgb.QuantileDMatrix(X, y, qid=qid, weight=w)
        Xy = xgb.DMatrix(X, y, qid=qid, weight=w)
        xgb.train({"tree_method": "hist", "objective": "rank:ndcg"}, Xy)

        from_qdm = xgb.QuantileDMatrix(X, weight=w, ref=Xy_qdm)
        from_dm = xgb.QuantileDMatrix(X, weight=w, ref=Xy)
        assert predictor_equal(from_qdm, from_dm)

    def test_check_inf(self) -> None:
        rng = np.random.default_rng(1994)
        check_inf(rng)

    # we don't test empty Quantile DMatrix in single node construction.
    @given(
        strategies.integers(1, 1000),
        strategies.integers(1, 100),
        strategies.fractions(0, 0.99),
    )
    @settings(deadline=None, print_blob=True)
    def test_to_csr(self, n_samples: int, n_features: int, sparsity: float) -> None:
        csr, y = make_sparse_regression(n_samples, n_features, sparsity, False)
        csr = csr.astype(np.float32)
        qdm = xgb.QuantileDMatrix(data=csr, label=y)
        ret = qdm.get_data()
        np.testing.assert_equal(csr.indptr, ret.indptr)
        np.testing.assert_equal(csr.indices, ret.indices)

        booster = xgb.train({"tree_method": "hist"}, dtrain=qdm)

        np.testing.assert_allclose(
            booster.predict(qdm), booster.predict(xgb.DMatrix(qdm.get_data()))
        )

    def test_dtypes(self) -> None:
        """Checks for both np array and pd DataFrame."""
        n_samples = 128
        n_features = 16
        for orig, x in np_dtypes(n_samples, n_features):
            m0 = xgb.QuantileDMatrix(orig)
            m1 = xgb.QuantileDMatrix(x)
            assert predictor_equal(m0, m1)

        # unsupported types
        for dtype in [
            np.bytes_,
            np.complex64,
            np.complex128,
        ]:
            X: np.ndarray = np.array(orig, dtype=dtype)
            with pytest.raises(ValueError):
                xgb.QuantileDMatrix(X)

    def test_changed_max_bin(self) -> None:
        n_samples = 128
        n_features = 16
        csr, y = make_sparse_regression(n_samples, n_features, 0.5, False)
        Xy = xgb.QuantileDMatrix(csr, y, max_bin=9)
        booster = xgb.train({"max_bin": 9}, Xy, num_boost_round=2)

        Xy = xgb.QuantileDMatrix(csr, y, max_bin=11)

        with pytest.raises(ValueError, match="consistent"):
            xgb.train({}, Xy, num_boost_round=2, xgb_model=booster)

    def test_mixed_sparsity(self) -> None:
        run_mixed_sparsity("cpu")
