import testing as tm
import unittest
import pytest
import xgboost as xgb
import numpy as np
from hypothesis import given, strategies, settings, note

exact_parameter_strategy = strategies.fixed_dictionaries({
    'nthread': strategies.integers(1, 4),
    'max_depth': strategies.integers(1, 11),
    'min_child_weight': strategies.floats(0.5, 2.0),
    'alpha': strategies.floats(0.0, 2.0),
    'lambda': strategies.floats(1e-5, 2.0),
    'eta': strategies.floats(0.01, 0.5),
    'gamma': strategies.floats(0.0, 2.0),
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


class TestTreeMethod(unittest.TestCase):
    @given(exact_parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=2000)
    def test_exact(self, param, num_rounds, dataset):
        param['tree_method'] = 'exact'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result['train'][dataset.metric])

    @given(exact_parameter_strategy, strategies.integers(1, 20),
           tm.dataset_strategy)
    @settings(deadline=2000)
    def test_approx(self, param, num_rounds, dataset):
        param['tree_method'] = 'approx'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result['train'][dataset.metric], 1e-3)

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
    @settings(deadline=2000)
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
