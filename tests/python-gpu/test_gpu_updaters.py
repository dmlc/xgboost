import sys
from typing import Any, Dict

import numpy as np
import pytest
from hypothesis import assume, given, note, settings, strategies

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.params import (
    cat_parameter_strategy,
    exact_parameter_strategy,
    hist_cache_strategy,
    hist_parameter_strategy,
)
from xgboost.testing.updater import (
    check_categorical_missing,
    check_categorical_ohe,
    check_get_quantile_cut,
    check_init_estimation,
    check_quantile_loss,
    train_result,
)

sys.path.append("tests/python")
import test_updaters as test_up

pytestmark = tm.timeout(30)


class TestGPUUpdatersMulti:
    @given(
        hist_parameter_strategy, strategies.integers(1, 20), tm.multi_dataset_strategy
    )
    @settings(deadline=None, max_examples=50, print_blob=True)
    def test_hist(self, param, num_rounds, dataset):
        param["tree_method"] = "gpu_hist"
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric])


class TestGPUUpdaters:
    cputest = test_up.TestTreeMethod()

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
    def test_sparse(self, dataset):
        param = {"tree_method": "hist", "max_bin": 64}
        hist_result = train_result(param, dataset.get_dmat(), 16)
        note(str(hist_result))
        assert tm.non_increasing(hist_result["train"][dataset.metric])

        param = {"tree_method": "gpu_hist", "max_bin": 64}
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
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_ohe(self, rows, cols, rounds, cats):
        check_categorical_ohe(
            rows=rows,
            cols=cols,
            rounds=rounds,
            cats=cats,
            device="cuda",
            tree_method="hist",
            extmem=False,
        )

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
        cat_parameters["tree_method"] = "gpu_hist"
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
    def test_max_cat(self) -> None:
        self.cputest.run_max_cat("gpu_hist")

    def test_categorical_32_cat(self):
        """32 hits the bound of integer bitset, so special test"""
        rows = 1000
        check_categorical_ohe(
            rows=rows,
            cols=10,
            rounds=4,
            cats=32,
            device="cuda",
            tree_method="hist",
            extmem=False,
        )

    @pytest.mark.skipif(**tm.no_cupy())
    def test_invalid_category(self):
        self.cputest.run_invalid_category("gpu_hist")

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
        param["tree_method"] = "gpu_hist"
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
    def test_external_memory(self, param, num_rounds, dataset):
        if dataset.name.endswith("-l1"):
            return
        # We cannot handle empty dataset yet
        assume(len(dataset.y) > 0)
        param["tree_method"] = "gpu_hist"
        param = dataset.set_params(param)
        m = dataset.get_external_dmat()
        external_result = train_result(param, m, num_rounds)
        del m
        assert tm.non_increasing(external_result["train"][dataset.metric])

    def test_empty_dmatrix_prediction(self):
        # FIXME(trivialfis): This should be done with all updaters
        kRows = 0
        kCols = 100

        X = np.empty((kRows, kCols))
        y = np.empty((kRows,))

        dtrain = xgb.DMatrix(X, y)

        bst = xgb.train(
            {"verbosity": 2, "tree_method": "gpu_hist", "gpu_id": 0},
            dtrain,
            verbose_eval=True,
            num_boost_round=6,
            evals=[(dtrain, "Train")],
        )

        kRows = 100
        X = np.random.randn(kRows, kCols)

        dtest = xgb.DMatrix(X)
        predictions = bst.predict(dtest)
        # non-distributed, 0.0 is returned due to base_score estimation with 0 gradient.
        np.testing.assert_allclose(predictions, 0.0, 1e-6)

    @pytest.mark.mgpu
    @given(tm.make_dataset_strategy(), strategies.integers(0, 10))
    @settings(deadline=None, max_examples=10, print_blob=True)
    def test_specified_gpu_id_gpu_update(self, dataset, gpu_id):
        param = {"tree_method": "gpu_hist", "gpu_id": gpu_id}
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), 10)
        assert tm.non_increasing(result["train"][dataset.metric])

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize("weighted", [True, False])
    def test_adaptive(self, weighted) -> None:
        self.cputest.run_adaptive("gpu_hist", weighted)

    def test_init_estimation(self) -> None:
        check_init_estimation("gpu_hist")

    @pytest.mark.parametrize("weighted", [True, False])
    def test_quantile_loss(self, weighted: bool) -> None:
        check_quantile_loss("gpu_hist", weighted)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_issue8824(self):
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
                "tree_method": "gpu_hist",
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
