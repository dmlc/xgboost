from itertools import product
from typing import Any, Dict

import numpy as np
import pytest
from hypothesis import given, note, settings, strategies

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
    run_adaptive,
    run_invalid_category,
    run_max_cat,
    train_result,
)


class TestTreeMethod:
    USE_ONEHOT = np.iinfo(np.int32).max
    USE_PART = 1

    @given(
        exact_parameter_strategy, strategies.integers(1, 20), tm.make_dataset_strategy()
    )
    @settings(deadline=None, print_blob=True)
    def test_exact(self, param, num_rounds, dataset):
        if dataset.name.endswith("-l1"):
            return
        param["tree_method"] = "exact"
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result["train"][dataset.metric])

    def test_exact_sample_by_node_error(self) -> None:
        X, y, w = tm.make_regression(128, 12, False)
        with pytest.raises(ValueError, match="column sample by node"):
            xgb.train(
                {"tree_method": "exact", "colsample_bynode": 0.999},
                xgb.DMatrix(X, y, weight=w),
            )

        xgb.train(
            {"tree_method": "exact", "colsample_bynode": 1.0},
            xgb.DMatrix(X, y, weight=w),
            num_boost_round=2,
        )

    @pytest.mark.parametrize("tree_method", ["approx", "hist"])
    def test_colsample_rng(self, tree_method: str) -> None:
        """Test rng has an effect on column sampling."""
        X, y, _ = tm.make_regression(128, 16, use_cupy=False)
        reg0 = xgb.XGBRegressor(
            n_estimators=2,
            colsample_bynode=0.5,
            random_state=42,
            tree_method=tree_method,
        )
        reg0.fit(X, y)

        reg1 = xgb.XGBRegressor(
            n_estimators=2,
            colsample_bynode=0.5,
            random_state=43,
            tree_method=tree_method,
        )
        reg1.fit(X, y)

        assert list(reg0.feature_importances_) != list(reg1.feature_importances_)

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        hist_cache_strategy,
        strategies.integers(1, 20),
        tm.make_dataset_strategy(),
    )
    @settings(deadline=None, print_blob=True)
    def test_approx(
        self,
        param: Dict[str, Any],
        hist_param: Dict[str, Any],
        cache_param: Dict[str, Any],
        num_rounds: int,
        dataset: tm.TestDataset,
    ) -> None:
        param["tree_method"] = "approx"
        param = dataset.set_params(param)
        param.update(hist_param)
        param.update(cache_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric])

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_pruner(self):
        import sklearn

        params = {"tree_method": "exact"}
        cancer = sklearn.datasets.load_breast_cancer()
        X = cancer["data"]
        y = cancer["target"]

        dtrain = xgb.DMatrix(X, y)
        booster = xgb.train(params, dtrain=dtrain, num_boost_round=10)
        grown = str(booster.get_dump())

        params = {"updater": "prune", "process_type": "update", "gamma": "0.2"}
        booster = xgb.train(
            params, dtrain=dtrain, num_boost_round=10, xgb_model=booster
        )
        after_prune = str(booster.get_dump())
        assert grown != after_prune

        booster = xgb.train(
            params, dtrain=dtrain, num_boost_round=10, xgb_model=booster
        )
        second_prune = str(booster.get_dump())
        # Second prune should not change the tree
        assert after_prune == second_prune

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        hist_cache_strategy,
        strategies.integers(1, 20),
        tm.make_dataset_strategy(),
    )
    @settings(deadline=None, print_blob=True)
    def test_hist(
        self,
        param: Dict[str, Any],
        hist_param: Dict[str, Any],
        cache_param: Dict[str, Any],
        num_rounds: int,
        dataset: tm.TestDataset,
    ) -> None:
        param["tree_method"] = "hist"
        param = dataset.set_params(param)
        param.update(hist_param)
        param.update(cache_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(str(result))
        assert tm.non_increasing(result["train"][dataset.metric])

    def test_hist_categorical(self):
        # hist must be same as exact on all-categorial data
        ag_dtrain, ag_dtest = tm.load_agaricus(__file__)
        ag_param = {
            "max_depth": 2,
            "tree_method": "hist",
            "eta": 1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
        }
        hist_res = {}
        exact_res = {}

        xgb.train(
            ag_param,
            ag_dtrain,
            10,
            evals=[(ag_dtrain, "train"), (ag_dtest, "test")],
            evals_result=hist_res,
        )
        ag_param["tree_method"] = "exact"
        xgb.train(
            ag_param,
            ag_dtrain,
            10,
            evals=[(ag_dtrain, "train"), (ag_dtest, "test")],
            evals_result=exact_res,
        )
        assert hist_res["train"]["auc"] == exact_res["train"]["auc"]
        assert hist_res["test"]["auc"] == exact_res["test"]["auc"]

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_hist_degenerate_case(self):
        # Test a degenerate case where the quantile sketcher won't return any
        # quantile points for a particular feature (the second feature in
        # this example). Source: https://github.com/dmlc/xgboost/issues/2943
        nan = np.nan
        param = {"missing": nan, "tree_method": "hist"}
        model = xgb.XGBRegressor(**param)
        X = np.array(
            [
                [6.18827160e05, 1.73000000e02],
                [6.37345679e05, nan],
                [6.38888889e05, nan],
                [6.28086420e05, nan],
            ]
        )
        y = [1000000.0, 0.0, 0.0, 500000.0]
        w = [0, 0, 1, 0]
        model.fit(X, y, sample_weight=w)

    @given(tm.sparse_datasets_strategy)
    @settings(deadline=None, print_blob=True)
    def test_sparse(self, dataset):
        param = {"tree_method": "hist", "max_bin": 64}
        hist_result = train_result(param, dataset.get_dmat(), 16)
        note(str(hist_result))
        assert tm.non_increasing(hist_result["train"][dataset.metric])

        param = {"tree_method": "approx", "max_bin": 64}
        approx_result = train_result(param, dataset.get_dmat(), 16)
        note(str(approx_result))
        assert tm.non_increasing(approx_result["train"][dataset.metric])

        np.testing.assert_allclose(
            hist_result["train"]["rmse"], approx_result["train"]["rmse"]
        )

    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    def test_invalid_category(self, tree_method: str) -> None:
        run_invalid_category(tree_method, "cpu")

    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    @pytest.mark.skipif(**tm.no_pandas())
    def test_max_cat(self, tree_method: str) -> None:
        run_max_cat(tree_method, "cpu")

    @given(
        strategies.integers(10, 400),
        strategies.integers(3, 8),
        strategies.integers(1, 2),
        strategies.integers(4, 7),
    )
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_ohe(
        self, rows: int, cols: int, rounds: int, cats: int
    ) -> None:
        check_categorical_ohe(
            rows=rows,
            cols=cols,
            rounds=rounds,
            cats=cats,
            device="cpu",
            tree_method="approx",
            extmem=False,
        )
        check_categorical_ohe(
            rows=rows,
            cols=cols,
            rounds=rounds,
            cats=cats,
            device="cpu",
            tree_method="hist",
            extmem=False,
        )

    @given(
        tm.categorical_dataset_strategy,
        exact_parameter_strategy,
        hist_parameter_strategy,
        cat_parameter_strategy,
        strategies.integers(4, 32),
        strategies.sampled_from(["hist", "approx"]),
    )
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical(
        self,
        dataset: tm.TestDataset,
        exact_parameters: Dict[str, Any],
        hist_parameters: Dict[str, Any],
        cat_parameters: Dict[str, Any],
        n_rounds: int,
        tree_method: str,
    ) -> None:
        cat_parameters.update(exact_parameters)
        cat_parameters.update(hist_parameters)
        cat_parameters["tree_method"] = tree_method

        results = train_result(cat_parameters, dataset.get_dmat(), n_rounds)
        tm.non_increasing(results["train"]["rmse"])

    @given(
        hist_parameter_strategy,
        cat_parameter_strategy,
        strategies.sampled_from(["hist", "approx"]),
    )
    @settings(deadline=None, print_blob=True)
    def test_categorical_ames_housing(
        self,
        hist_parameters: Dict[str, Any],
        cat_parameters: Dict[str, Any],
        tree_method: str,
    ) -> None:
        cat_parameters.update(hist_parameters)
        dataset = tm.TestDataset(
            "ames_housing", tm.data.get_ames_housing, "reg:squarederror", "rmse"
        )
        cat_parameters["tree_method"] = tree_method
        results = train_result(cat_parameters, dataset.get_dmat(), 16)
        tm.non_increasing(results["train"]["rmse"])

    @given(
        strategies.integers(10, 400),
        strategies.integers(3, 8),
        strategies.integers(4, 7),
    )
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_missing(self, rows: int, cols: int, cats: int) -> None:
        check_categorical_missing(
            rows, cols, cats, device="cpu", tree_method="approx", extmem=False
        )
        check_categorical_missing(
            rows, cols, cats, device="cpu", tree_method="hist", extmem=False
        )

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize(
        "tree_method,weighted", list(product(["approx", "hist"], [True, False]))
    )
    def test_adaptive(self, tree_method: str, weighted: bool) -> None:
        run_adaptive(tree_method, weighted, "cpu")

    def test_init_estimation(self) -> None:
        check_init_estimation("hist", "cpu")

    @pytest.mark.parametrize("weighted", [True, False])
    def test_quantile_loss(self, weighted: bool) -> None:
        check_quantile_loss("hist", weighted, "cpu")

    @pytest.mark.skipif(**tm.no_pandas())
    @pytest.mark.parametrize("tree_method", ["hist"])
    def test_get_quantile_cut(self, tree_method: str) -> None:
        check_get_quantile_cut(tree_method, "cpu")
