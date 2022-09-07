from random import choice
from string import ascii_lowercase
import testing as tm
import pytest
import xgboost as xgb
import numpy as np
from hypothesis import given, strategies, settings, note

exact_parameter_strategy = strategies.fixed_dictionaries({
    'nthread': strategies.integers(1, 4),
    'max_depth': strategies.integers(1, 11),
    'min_child_weight': strategies.floats(0.5, 2.0),
    'alpha': strategies.floats(1e-5, 2.0),
    'lambda': strategies.floats(1e-5, 2.0),
    'eta': strategies.floats(0.01, 0.5),
    'gamma': strategies.floats(1e-5, 2.0),
    'seed': strategies.integers(0, 10),
    # We cannot enable subsampling as the training loss can increase
    # 'subsample': strategies.floats(0.5, 1.0),
    'colsample_bytree': strategies.floats(0.5, 1.0),
    'colsample_bylevel': strategies.floats(0.5, 1.0),
})

hist_parameter_strategy = strategies.fixed_dictionaries({
    'max_depth': strategies.integers(1, 11),
    'max_leaves': strategies.integers(0, 1024),
    'max_bin': strategies.integers(2, 512),
    'grow_policy': strategies.sampled_from(['lossguide', 'depthwise']),
}).filter(lambda x: (x['max_depth'] > 0 or x['max_leaves'] > 0) and (
    x['max_depth'] > 0 or x['grow_policy'] == 'lossguide'))


def train_result(param, dmat, num_rounds):
    result = {}
    xgb.train(param, dmat, num_rounds, [(dmat, 'train')], verbose_eval=False,
              evals_result=result)
    return result


class TestTreeMethod:
    @given(exact_parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=None, print_blob=True)
    def test_exact(self, param, num_rounds, dataset):
        param['tree_method'] = 'exact'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result['train'][dataset.metric])

    @given(
        exact_parameter_strategy,
        hist_parameter_strategy,
        strategies.integers(1, 20),
        tm.dataset_strategy,
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

    @given(exact_parameter_strategy, hist_parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=None, print_blob=True)
    def test_hist(self, param, hist_param, num_rounds, dataset):
        param['tree_method'] = 'hist'
        param = dataset.set_params(param)
        param.update(hist_param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result['train'][dataset.metric])

    def test_hist_categorical(self):
        # hist must be same as exact on all-categorial data
        dpath = 'demo/data/'
        ag_dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        ag_dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
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
                Xy = xgb.DeviceQuantileDMatrix(X, y, feature_types=["c"] * 10)

    def test_invalid_category(self) -> None:
        self.run_invalid_category("approx")
        self.run_invalid_category("hist")

    def run_max_cat(self, tree_method: str) -> None:
        """Test data with size smaller than number of categories."""
        import pandas as pd
        n_cat = 100
        n = 5
        X = pd.Series(
            ["".join(choice(ascii_lowercase) for i in range(3)) for i in range(n_cat)],
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
    def test_max_cat(self, tree_method) -> None:
        self.run_max_cat(tree_method)

    def run_categorical_basic(self, rows, cols, rounds, cats, tree_method):
        onehot, label = tm.make_categorical(rows, cols, cats, True)
        cat, _ = tm.make_categorical(rows, cols, cats, False)

        by_etl_results = {}
        by_builtin_results = {}

        predictor = "gpu_predictor" if tree_method == "gpu_hist" else None
        # Use one-hot exclusively
        parameters = {
            "tree_method": tree_method, "predictor": predictor, "max_cat_to_onehot": 9999
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

        # There are guidelines on how to specify tolerance based on considering output as
        # random variables. But in here the tree construction is extremely sensitive to
        # floating point errors. An 1e-5 error in a histogram bin can lead to an entirely
        # different tree.  So even though the test is quite lenient, hypothesis can still
        # pick up falsifying examples from time to time.
        np.testing.assert_allclose(
            np.array(by_etl_results["Train"]["rmse"]),
            np.array(by_builtin_results["Train"]["rmse"]),
            rtol=1e-3,
        )
        assert tm.non_increasing(by_builtin_results["Train"]["rmse"])

        by_grouping: xgb.callback.TrainingCallback.EvalsLog = {}
        parameters["max_cat_to_onehot"] = 1
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
    def test_categorical(self, rows, cols, rounds, cats):
        self.run_categorical_basic(rows, cols, rounds, cats, "approx")
        self.run_categorical_basic(rows, cols, rounds, cats, "hist")
