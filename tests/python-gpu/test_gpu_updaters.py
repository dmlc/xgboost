from typing import Any, Dict

import numpy as np
import pytest
import xgboost as xgb
from hypothesis import assume, given, note, settings, strategies
from xgboost import testing as tm
from xgboost.testing.params import (
    cat_parameter_strategy,
    exact_parameter_strategy,
    hist_cache_strategy,
    hist_parameter_strategy,
)
from xgboost.testing.updater import (
    check_categorical_mixed,
    check_categorical_missing,
    check_categorical_ohe,
    check_get_quantile_cut,
    check_quantile_loss,
    run_invalid_category,
    run_max_cat,
    train_result,
)

pytestmark = tm.timeout(30)


class TestGPUUpdaters:
    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        hist_cache_strategy,
        strategies.integers(1, 20),
        tm.make_dataset_strategy(),
    )
    @settings(deadline=None, max_examples=50, print_blob=True)
    def test_gpu_hist(
        self,
        param: Dict[str, Any],
        hist_param: Dict[str, Any],
        cache_param: Dict[str, Any],
        num_rounds: int,
        dataset: tm.TestDataset,
    ) -> None:
        param.update({"tree_method": "hist", "device": "cuda"})
        param.update(hist_param)
        param.update(cache_param)
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric])

    @pytest.mark.parametrize("tree_method", ["approx", "hist"])
    def test_cache_size(self, tree_method: str) -> None:
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=4096, n_features=64, random_state=1994)
        Xy = xgb.DMatrix(X, y)
        results = []
        for cache_size in [1, 3, 2048]:
            params: Dict[str, Any] = {"tree_method": tree_method, "device": "cuda"}
            params["max_cached_hist_node"] = cache_size
            evals_result: Dict[str, Dict[str, list]] = {}
            xgb.train(
                params,
                Xy,
                num_boost_round=4,
                evals=[(Xy, "Train")],
                evals_result=evals_result,
            )
            results.append(evals_result["Train"]["rmse"])
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i])

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        hist_cache_strategy,
        strategies.integers(1, 20),
        tm.make_dataset_strategy(),
    )
    @settings(deadline=None, print_blob=True)
    def test_gpu_approx(
        self,
        param: Dict[str, Any],
        hist_param: Dict[str, Any],
        cache_param: Dict[str, Any],
        num_rounds: int,
        dataset: tm.TestDataset,
    ) -> None:
        param.update({"tree_method": "approx", "device": "cuda"})
        param.update(hist_param)
        param.update(cache_param)
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric])

    @given(tm.sparse_datasets_strategy)
    @settings(deadline=None, print_blob=True)
    def test_sparse(self, dataset: tm.TestDataset) -> None:
        param = {"tree_method": "hist", "max_bin": 64}
        hist_result = train_result(param, dataset.get_dmat(), 16)
        note(str(hist_result))
        assert tm.non_increasing(hist_result["train"][dataset.metric])

        param = {"tree_method": "hist", "max_bin": 64, "device": "cuda"}
        gpu_hist_result = train_result(param, dataset.get_dmat(), 16)
        note(str(gpu_hist_result))
        assert tm.non_increasing(gpu_hist_result["train"][dataset.metric])

        np.testing.assert_allclose(
            hist_result["train"]["rmse"], gpu_hist_result["train"]["rmse"], rtol=1e-2
        )

    @given(
        strategies.integers(10, 400),
        strategies.integers(3, 8),
        strategies.integers(1, 2),
        strategies.integers(4, 7),
        strategies.integers(5, 16),
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_ohe(
        self, rows: int, cols: int, rounds: int, cats: int, max_bin: int
    ) -> None:
        check_categorical_ohe(
            rows=rows,
            cols=cols,
            rounds=rounds,
            cats=cats,
            device="cuda",
            tree_method="hist",
            extmem=False,
            max_bin=max_bin,
        )
        check_categorical_ohe(
            rows=rows,
            cols=cols,
            rounds=rounds,
            cats=cats,
            device="cuda",
            tree_method="hist",
            extmem=False,
            multi_target=True,
            max_bin=max_bin,
        )

    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_mixed(self) -> None:
        check_categorical_mixed("cuda")

    @given(
        tm.categorical_dataset_strategy,
        hist_parameter_strategy,
        cat_parameter_strategy,
        strategies.integers(4, 32),
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_hist(
        self,
        dataset: tm.TestDataset,
        hist_parameters: Dict[str, Any],
        cat_parameters: Dict[str, Any],
        n_rounds: int,
    ) -> None:
        cat_parameters.update(hist_parameters)
        cat_parameters["tree_method"] = "hist"
        cat_parameters["device"] = "cuda"

        results = train_result(cat_parameters, dataset.get_dmat(), n_rounds)
        tm.non_increasing(results["train"]["rmse"])

    @given(
        tm.categorical_dataset_strategy,
        hist_parameter_strategy,
        cat_parameter_strategy,
        strategies.integers(4, 32),
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_approx(
        self,
        dataset: tm.TestDataset,
        hist_parameters: Dict[str, Any],
        cat_parameters: Dict[str, Any],
        n_rounds: int,
    ) -> None:
        cat_parameters.update(hist_parameters)
        cat_parameters["tree_method"] = "approx"
        cat_parameters["device"] = "cuda"

        results = train_result(cat_parameters, dataset.get_dmat(), n_rounds)
        tm.non_increasing(results["train"]["rmse"])

    @given(
        hist_parameter_strategy,
        cat_parameter_strategy,
    )
    @settings(deadline=None, max_examples=10, print_blob=True)
    def test_categorical_ames_housing(
        self,
        hist_parameters: Dict[str, Any],
        cat_parameters: Dict[str, Any],
    ) -> None:
        cat_parameters.update(hist_parameters)
        dataset = tm.TestDataset(
            "ames_housing", tm.data.get_ames_housing, "reg:squarederror", "rmse"
        )
        cat_parameters["tree_method"] = "hist"
        cat_parameters["device"] = "cuda"
        results = train_result(cat_parameters, dataset.get_dmat(), 16)
        tm.non_increasing(results["train"]["rmse"])

    @given(
        strategies.integers(10, 400),
        strategies.integers(3, 8),
        strategies.integers(4, 7),
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_missing(self, rows: int, cols: int, cats: int) -> None:
        check_categorical_missing(
            rows, cols, cats, device="cuda", tree_method="approx", extmem=False
        )
        check_categorical_missing(
            rows, cols, cats, device="cuda", tree_method="hist", extmem=False
        )

    @pytest.mark.skipif(**tm.no_pandas())
    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_max_cat(self, tree_method: str) -> None:
        run_max_cat(tree_method, "cuda")

    @pytest.mark.parametrize("cats", [32, 64])
    @pytest.mark.parametrize("multi_target", [False, True])
    def test_categorical_bitfield_boundaries(
        self, cats: int, multi_target: bool
    ) -> None:
        """Test scalar and vector categorical splits at bit-field word boundaries."""
        n_targets = 3 if multi_target else 1
        X, y = tm.make_categorical(
            1000, 2, cats, onehot=False, n_targets=n_targets, sparsity=0.0
        )
        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        params: Dict[str, Any] = {"device": "cuda", "tree_method": "hist"}
        if multi_target:
            params["multi_strategy"] = "multi_output_tree"

        for max_cat_to_onehot in [1, 128]:
            params["max_cat_to_onehot"] = max_cat_to_onehot
            booster = xgb.train(params, Xy, num_boost_round=1)

            assert booster.get_score(importance_type="weight")
            predt = booster.predict(Xy)
            assert predt.shape == y.shape
            assert np.isfinite(predt).all()

    @pytest.mark.skipif(**tm.no_cupy())
    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_invalid_category(self, tree_method: str) -> None:
        run_invalid_category(tree_method, "cuda")

    @pytest.mark.skipif(**tm.no_cupy())
    @given(
        hist_parameter_strategy,
        strategies.integers(1, 20),
        tm.make_dataset_strategy(),
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_gpu_hist_device_dmatrix(
        self, param: dict, num_rounds: int, dataset: tm.TestDataset
    ) -> None:
        # We cannot handle empty dataset yet
        assume(len(dataset.y) > 0)
        param["tree_method"] = "hist"
        param["device"] = "cuda"
        param = dataset.set_params(param)
        result = train_result(
            param,
            dataset.get_device_dmat(max_bin=param.get("max_bin", None)),
            num_rounds,
        )
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric], tolerance=1e-3)

    @given(
        hist_parameter_strategy,
        strategies.integers(1, 3),
        tm.make_dataset_strategy(),
    )
    @settings(deadline=None, max_examples=10, print_blob=True)
    def test_external_memory(
        self, param: Dict[str, Any], num_rounds: int, dataset: tm.TestDataset
    ) -> None:
        # We cannot handle empty dataset yet
        assume(len(dataset.y) > 0)

        with xgb.config_context(use_rmm=True):
            param["tree_method"] = "hist"
            param["device"] = "cuda"
            param = dataset.set_params(param)
            m = dataset.get_external_dmat()
            external_result = train_result(param, m, num_rounds)
            del m
            assert tm.non_increasing(external_result["train"][dataset.metric])

    def test_empty_dmatrix_prediction(self) -> None:
        # FIXME(trivialfis): This should be done with all updaters
        kRows = 0
        kCols = 100

        X = np.empty((kRows, kCols))
        y = np.empty((kRows,))

        dtrain = xgb.DMatrix(X, y)

        bst = xgb.train(
            {"verbosity": 2, "tree_method": "hist", "device": "cuda"},
            dtrain,
            verbose_eval=True,
            num_boost_round=6,
            evals=[(dtrain, "Train")],
        )

        kRows = 100
        X_test = np.random.randn(kRows, kCols)

        dtest = xgb.DMatrix(X_test)
        predictions = bst.predict(dtest)
        # non-distributed, 0.0 is returned due to base_score estimation with 0 gradient.
        np.testing.assert_allclose(predictions, 0.0, 1e-6)

    @pytest.mark.mgpu
    @given(tm.make_dataset_strategy(), strategies.integers(0, 10))
    @settings(deadline=None, max_examples=10, print_blob=True)
    def test_specified_gpu_id_gpu_update(
        self, dataset: tm.TestDataset, gpu_id: int
    ) -> None:
        param = {"tree_method": "hist", "device": f"cuda:{gpu_id}"}
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), 10)
        assert tm.non_increasing(result["train"][dataset.metric])

    @pytest.mark.parametrize("weighted", [True, False])
    def test_quantile_loss(self, weighted: bool) -> None:
        check_quantile_loss("hist", weighted, "cuda")

    @pytest.mark.skipif(**tm.no_pandas())
    def test_issue8824(self) -> None:
        # column sampling by node crashes because shared pointers go out of scope
        import pandas as pd

        data = pd.DataFrame(np.random.rand(1024, 8))
        data.columns = "x" + data.columns.astype(str)
        features = data.columns
        data["y"] = data.sum(axis=1) < 4
        dtrain = xgb.DMatrix(data[features], label=data["y"])
        model = xgb.train(
            dtrain=dtrain,
            params={
                "max_depth": 5,
                "learning_rate": 0.05,
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "colsample_bytree": 0.5,
                "colsample_bylevel": 0.5,
                "colsample_bynode": 0.5,  # Causes issues
                "reg_alpha": 0.05,
                "reg_lambda": 0.005,
                "seed": 66,
                "subsample": 0.5,
                "gamma": 0.2,
                "eval_metric": "auc",
            },
            num_boost_round=150,
        )

    @pytest.mark.skipif(**tm.no_cudf())
    def test_get_quantile_cut(self) -> None:
        check_get_quantile_cut("hist", "cuda")
