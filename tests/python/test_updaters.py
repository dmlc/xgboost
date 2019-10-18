import testing as tm
import unittest
import pytest
import xgboost as xgb
import numpy as np

try:
    from regression_test_utilities import run_suite, parameter_combinations, \
        assert_results_non_increasing
except ImportError:
    None


class TestUpdaters(unittest.TestCase):
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_histmaker(self):
        variable_param = {'updater': ['grow_histmaker'], 'max_depth': [2, 8]}
        for param in parameter_combinations(variable_param):
            result = run_suite(param)
            assert_results_non_increasing(result, 1e-2)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_colmaker(self):
        variable_param = {'updater': ['grow_colmaker'], 'max_depth': [2, 8]}
        for param in parameter_combinations(variable_param):
            result = run_suite(param)
            assert_results_non_increasing(result, 1e-2)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_fast_histmaker(self):
        variable_param = {'tree_method': ['hist'],
                          'max_depth': [2, 8],
                          'max_bin': [2, 256],
                          'grow_policy': ['depthwise', 'lossguide'],
                          'max_leaves': [64, 0],
                          'verbosity': [0]}
        for param in parameter_combinations(variable_param):
            result = run_suite(param)
            assert_results_non_increasing(result, 1e-2)

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
    def test_fast_histmaker_degenerate_case(self):
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
