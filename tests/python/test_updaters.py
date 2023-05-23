import json
from string import ascii_lowercase
from typing import Any, Dict, List

import numpy as np
import pytest
from hypothesis import given, note, settings, strategies

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.params import (
    cat_parameter_strategy,
    exact_parameter_strategy,
    hist_multi_parameter_strategy,
    hist_parameter_strategy,
)
from xgboost.testing.updater import check_init_estimation, check_quantile_loss


def train_result(param, dmat, num_rounds):
    result = {}
    booster = xgb.train(
        param,
        dmat,
        num_rounds,
        [(dmat, "train")],
        verbose_eval=False,
        evals_result=result,
    )
    assert booster.num_features() == dmat.num_col()
    assert booster.num_boosted_rounds() == num_rounds
    assert booster.feature_names == dmat.feature_names
    assert booster.feature_types == dmat.feature_types

    return result


class TestTreeMethodMulti:
    @given(
        exact_parameter_strategy, strategies.integers(1, 20), tm.multi_dataset_strategy
    )
    @settings(deadline=None, print_blob=True)
    def test_exact(self, param: dict, num_rounds: int, dataset: tm.TestDataset) -> None:
        if dataset.name.endswith("-l1"):
            return
        param["tree_method"] = "exact"
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result["train"][dataset.metric])

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        strategies.integers(1, 20),
        tm.multi_dataset_strategy,
    )
    @settings(deadline=None, print_blob=True)
    def test_approx(self, param, hist_param, num_rounds, dataset):
        param["tree_method"] = "approx"
        param = dataset.set_params(param)
        param.update(hist_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result["train"][dataset.metric])

    @given(
        exact_parameter_strategy,
        hist_multi_parameter_strategy,
        strategies.integers(1, 20),
        tm.multi_dataset_strategy,
    )
    @settings(deadline=None, print_blob=True)
    def test_hist(
        self, param: dict, hist_param: dict, num_rounds: int, dataset: tm.TestDataset
    ) -> None:
        if dataset.name.endswith("-l1"):
            return
        param["tree_method"] = "hist"
        param = dataset.set_params(param)
        param.update(hist_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result["train"][dataset.metric])


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
        param['tree_method'] = 'exact'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result['train'][dataset.metric])

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        strategies.integers(1, 20),
        tm.make_dataset_strategy(),
    )
    @settings(deadline=None, print_blob=True)
    def test_approx(self, param, hist_param, num_rounds, dataset):
        param["tree_method"] = "approx"
        param = dataset.set_params(param)
        param.update(hist_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result["train"][dataset.metric])

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_pruner(self):
        import sklearn
        params = {'tree_method': 'exact'}
        cancer = sklearn.datasets.load_breast_cancer()
        X = cancer['data']
        y = cancer["target"]

        dtrain = xgb.DMatrix(X, y)
        booster = xgb.train(params, dtrain=dtrain, num_boost_round=10)
        grown = str(booster.get_dump())

        params = {'updater': 'prune', 'process_type': 'update', 'gamma': '0.2'}
        booster = xgb.train(params, dtrain=dtrain, num_boost_round=10,
                            xgb_model=booster)
        after_prune = str(booster.get_dump())
        assert grown != after_prune

        booster = xgb.train(params, dtrain=dtrain, num_boost_round=10,
                            xgb_model=booster)
        second_prune = str(booster.get_dump())
        # Second prune should not change the tree
        assert after_prune == second_prune

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        strategies.integers(1, 20),
        tm.make_dataset_strategy()
    )
    @settings(deadline=None, print_blob=True)
    def test_hist(self, param: dict, hist_param: dict, num_rounds: int, dataset: tm.TestDataset) -> None:
        param['tree_method'] = 'hist'
        param = dataset.set_params(param)
        param.update(hist_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result['train'][dataset.metric])

    def test_hist_categorical(self):
        # hist must be same as exact on all-categorial data
        ag_dtrain, ag_dtest = tm.load_agaricus(__file__)
        ag_param = {'max_depth': 2,
                    'tree_method': 'hist',
                    'eta': 1,
                    'verbosity': 0,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc'}
        hist_res = {}
        exact_res = {}

        xgb.train(ag_param, ag_dtrain, 10,
                  [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                  evals_result=hist_res)
        ag_param["tree_method"] = "exact"
        xgb.train(ag_param, ag_dtrain, 10,
                  [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                  evals_result=exact_res)
        assert hist_res['train']['auc'] == exact_res['train']['auc']
        assert hist_res['test']['auc'] == exact_res['test']['auc']

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_hist_degenerate_case(self):
        # Test a degenerate case where the quantile sketcher won't return any
        # quantile points for a particular feature (the second feature in
        # this example). Source: https://github.com/dmlc/xgboost/issues/2943
        nan = np.nan
        param = {'missing': nan, 'tree_method': 'hist'}
        model = xgb.XGBRegressor(**param)
        X = np.array([[6.18827160e+05, 1.73000000e+02], [6.37345679e+05, nan],
                      [6.38888889e+05, nan], [6.28086420e+05, nan]])
        y = [1000000., 0., 0., 500000.]
        w = [0, 0, 1, 0]
        model.fit(X, y, sample_weight=w)

    @given(tm.sparse_datasets_strategy)
    @settings(deadline=None, print_blob=True)
    def test_sparse(self, dataset):
        param = {"tree_method": "hist", "max_bin": 64}
        hist_result = train_result(param, dataset.get_dmat(), 16)
        note(hist_result)
        assert tm.non_increasing(hist_result['train'][dataset.metric])

        param = {"tree_method": "approx", "max_bin": 64}
        approx_result = train_result(param, dataset.get_dmat(), 16)
        note(approx_result)
        assert tm.non_increasing(approx_result['train'][dataset.metric])

        np.testing.assert_allclose(
            hist_result["train"]["rmse"], approx_result["train"]["rmse"]
        )

    def run_invalid_category(self, tree_method: str) -> None:
        rng = np.random.default_rng()
        # too large
        X = rng.integers(low=0, high=4, size=1000).reshape(100, 10)
        y = rng.normal(loc=0, scale=1, size=100)
        X[13, 7] = np.iinfo(np.int32).max + 1

        # Check is performed during sketching.
        Xy = xgb.DMatrix(X, y, feature_types=["c"] * 10)
        with pytest.raises(ValueError):
            xgb.train({"tree_method": tree_method}, Xy)

        X[13, 7] = 16777216
        Xy = xgb.DMatrix(X, y, feature_types=["c"] * 10)
        with pytest.raises(ValueError):
            xgb.train({"tree_method": tree_method}, Xy)

        # mixed positive and negative values
        X = rng.normal(loc=0, scale=1, size=1000).reshape(100, 10)
        y = rng.normal(loc=0, scale=1, size=100)

        Xy = xgb.DMatrix(X, y, feature_types=["c"] * 10)
        with pytest.raises(ValueError):
            xgb.train({"tree_method": tree_method}, Xy)

        if tree_method == "gpu_hist":
            import cupy as cp

            X, y = cp.array(X), cp.array(y)
            with pytest.raises(ValueError):
                Xy = xgb.QuantileDMatrix(X, y, feature_types=["c"] * 10)

    def test_invalid_category(self) -> None:
        self.run_invalid_category("approx")
        self.run_invalid_category("hist")

    def run_max_cat(self, tree_method: str) -> None:
        """Test data with size smaller than number of categories."""
        import pandas as pd

        rng = np.random.default_rng(0)
        n_cat = 100
        n = 5

        X = pd.Series(
            ["".join(rng.choice(list(ascii_lowercase), size=3)) for i in range(n_cat)],
            dtype="category",
        )[:n].to_frame()

        reg = xgb.XGBRegressor(
            enable_categorical=True,
            tree_method=tree_method,
            n_estimators=10,
        )
        y = pd.Series(range(n))
        reg.fit(X=X, y=y, eval_set=[(X, y)])
        assert tm.non_increasing(reg.evals_result()["validation_0"]["rmse"])

    @pytest.mark.parametrize("tree_method", ["hist", "approx"])
    @pytest.mark.skipif(**tm.no_pandas())
    def test_max_cat(self, tree_method) -> None:
        self.run_max_cat(tree_method)

    def run_categorical_missing(
        self, rows: int, cols: int, cats: int, tree_method: str
    ) -> None:
        parameters: Dict[str, Any] = {"tree_method": tree_method}
        cat, label = tm.make_categorical(
            n_samples=rows, n_features=cols, n_categories=cats, onehot=False, sparsity=0.5
        )
        Xy = xgb.DMatrix(cat, label, enable_categorical=True)

        def run(max_cat_to_onehot: int):
            # Test with onehot splits
            parameters["max_cat_to_onehot"] = max_cat_to_onehot

            evals_result: Dict[str, Dict] = {}
            booster = xgb.train(
                parameters,
                Xy,
                num_boost_round=16,
                evals=[(Xy, "Train")],
                evals_result=evals_result
            )
            assert tm.non_increasing(evals_result["Train"]["rmse"])
            y_predt = booster.predict(Xy)

            rmse = tm.root_mean_square(label, y_predt)
            np.testing.assert_allclose(rmse, evals_result["Train"]["rmse"][-1])

        # Test with OHE split
        run(self.USE_ONEHOT)

        # Test with partition-based split
        run(self.USE_PART)

    def run_categorical_ohe(
        self, rows: int, cols: int, rounds: int, cats: int, tree_method: str
    ) -> None:
        onehot, label = tm.make_categorical(rows, cols, cats, True)
        cat, _ = tm.make_categorical(rows, cols, cats, False)

        by_etl_results: Dict[str, Dict[str, List[float]]] = {}
        by_builtin_results: Dict[str, Dict[str, List[float]]] = {}

        predictor = "gpu_predictor" if tree_method == "gpu_hist" else None
        parameters: Dict[str, Any] = {
            "tree_method": tree_method,
            "predictor": predictor,
            # Use one-hot exclusively
            "max_cat_to_onehot": self.USE_ONEHOT
        }

        m = xgb.DMatrix(onehot, label, enable_categorical=False)
        xgb.train(
            parameters,
            m,
            num_boost_round=rounds,
            evals=[(m, "Train")],
            evals_result=by_etl_results,
        )

        m = xgb.DMatrix(cat, label, enable_categorical=True)
        xgb.train(
            parameters,
            m,
            num_boost_round=rounds,
            evals=[(m, "Train")],
            evals_result=by_builtin_results,
        )

        # There are guidelines on how to specify tolerance based on considering output
        # as random variables. But in here the tree construction is extremely sensitive
        # to floating point errors. An 1e-5 error in a histogram bin can lead to an
        # entirely different tree. So even though the test is quite lenient, hypothesis
        # can still pick up falsifying examples from time to time.
        np.testing.assert_allclose(
            np.array(by_etl_results["Train"]["rmse"]),
            np.array(by_builtin_results["Train"]["rmse"]),
            rtol=1e-3,
        )
        assert tm.non_increasing(by_builtin_results["Train"]["rmse"])

        by_grouping: Dict[str, Dict[str, List[float]]] = {}
        # switch to partition-based splits
        parameters["max_cat_to_onehot"] = self.USE_PART
        parameters["reg_lambda"] = 0
        m = xgb.DMatrix(cat, label, enable_categorical=True)
        xgb.train(
            parameters,
            m,
            num_boost_round=rounds,
            evals=[(m, "Train")],
            evals_result=by_grouping,
        )
        rmse_oh = by_builtin_results["Train"]["rmse"]
        rmse_group = by_grouping["Train"]["rmse"]
        # always better or equal to onehot when there's no regularization.
        for a, b in zip(rmse_oh, rmse_group):
            assert a >= b

        parameters["reg_lambda"] = 1.0
        by_grouping = {}
        xgb.train(
            parameters,
            m,
            num_boost_round=32,
            evals=[(m, "Train")],
            evals_result=by_grouping,
        )
        assert tm.non_increasing(by_grouping["Train"]["rmse"]), by_grouping

    @given(strategies.integers(10, 400), strategies.integers(3, 8),
           strategies.integers(1, 2), strategies.integers(4, 7))
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_ohe(
        self, rows: int, cols: int, rounds: int, cats: int
    ) -> None:
        self.run_categorical_ohe(rows, cols, rounds, cats, "approx")
        self.run_categorical_ohe(rows, cols, rounds, cats, "hist")

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
        strategies.integers(4, 7)
    )
    @settings(deadline=None, print_blob=True)
    @pytest.mark.skipif(**tm.no_pandas())
    def test_categorical_missing(self, rows, cols, cats):
        self.run_categorical_missing(rows, cols, cats, "approx")
        self.run_categorical_missing(rows, cols, cats, "hist")

    def run_adaptive(self, tree_method, weighted) -> None:
        rng = np.random.RandomState(1994)
        from sklearn.datasets import make_regression
        from sklearn.utils import stats

        n_samples = 256
        X, y = make_regression(n_samples, 16, random_state=rng)
        if weighted:
            w = rng.normal(size=n_samples)
            w -= w.min()
            Xy = xgb.DMatrix(X, y, weight=w)
            base_score = stats._weighted_percentile(y, w, percentile=50)
        else:
            Xy = xgb.DMatrix(X, y)
            base_score = np.median(y)

        booster_0 = xgb.train(
            {
                "tree_method": tree_method,
                "base_score": base_score,
                "objective": "reg:absoluteerror",
            },
            Xy,
            num_boost_round=1,
        )
        booster_1 = xgb.train(
            {"tree_method": tree_method, "objective": "reg:absoluteerror"},
            Xy,
            num_boost_round=1,
        )
        config_0 = json.loads(booster_0.save_config())
        config_1 = json.loads(booster_1.save_config())

        def get_score(config: Dict) -> float:
            return float(config["learner"]["learner_model_param"]["base_score"])

        assert get_score(config_0) == get_score(config_1)

        raw_booster = booster_1.save_raw(raw_format="deprecated")
        booster_2 = xgb.Booster(model_file=raw_booster)
        config_2 = json.loads(booster_2.save_config())
        assert get_score(config_1) == get_score(config_2)

        raw_booster = booster_1.save_raw(raw_format="ubj")
        booster_2 = xgb.Booster(model_file=raw_booster)
        config_2 = json.loads(booster_2.save_config())
        assert get_score(config_1) == get_score(config_2)

        booster_0 = xgb.train(
            {
                "tree_method": tree_method,
                "base_score": base_score + 1.0,
                "objective": "reg:absoluteerror",
            },
            Xy,
            num_boost_round=1,
        )
        config_0 = json.loads(booster_0.save_config())
        np.testing.assert_allclose(get_score(config_0), get_score(config_1) + 1)

        evals_result: Dict[str, Dict[str, list]] = {}
        xgb.train(
            {
                "tree_method": tree_method,
                "objective": "reg:absoluteerror",
                "subsample": 0.8,
                "eta": 1.0,
            },
            Xy,
            num_boost_round=10,
            evals=[(Xy, "Train")],
            evals_result=evals_result,
        )
        mae = evals_result["Train"]["mae"]
        assert mae[-1] < 20.0
        assert tm.non_increasing(mae)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize(
        "tree_method,weighted", [
            ("approx", False), ("hist", False), ("approx", True), ("hist", True)
        ]
    )
    def test_adaptive(self, tree_method, weighted) -> None:
        self.run_adaptive(tree_method, weighted)

    def test_init_estimation(self) -> None:
        check_init_estimation("hist")

    @pytest.mark.parametrize("weighted", [True, False])
    def test_quantile_loss(self, weighted: bool) -> None:
        check_quantile_loss("hist", weighted)
