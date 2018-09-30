import testing as tm
import unittest
import xgboost as xgb

try:
    from regression_test_utilities import run_suite, parameter_combinations, \
        assert_results_non_increasing
except ImportError:
    None


class TestUpdaters(unittest.TestCase):
    def test_histmaker(self):
        tm._skip_if_no_sklearn()
        variable_param = {'updater': ['grow_histmaker'], 'max_depth': [2, 8]}
        for param in parameter_combinations(variable_param):
            result = run_suite(param)
            assert_results_non_increasing(result, 1e-2)

    def test_colmaker(self):
        tm._skip_if_no_sklearn()
        variable_param = {'updater': ['grow_colmaker'], 'max_depth': [2, 8]}
        for param in parameter_combinations(variable_param):
            result = run_suite(param)
            assert_results_non_increasing(result, 1e-2)

    def test_fast_histmaker(self):
        tm._skip_if_no_sklearn()
        variable_param = {'tree_method': ['hist'], 'max_depth': [2, 8], 'max_bin': [2, 256],
                          'grow_policy': ['depthwise', 'lossguide'], 'max_leaves': [64, 0],
                          'silent': [1]}
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
                    'silent': 1,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc'}
        hist_res = {}
        exact_res = {}

        xgb.train(ag_param, ag_dtrain, 10, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                  evals_result=hist_res)
        ag_param["tree_method"] = "exact"
        xgb.train(ag_param, ag_dtrain, 10, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                  evals_result=exact_res)
        assert hist_res['train']['auc'] == exact_res['train']['auc']
        assert hist_res['test']['auc'] == exact_res['test']['auc']
