import xgboost as xgb
import testing as tm
import numpy as np
import unittest

rng = np.random.RandomState(1994)


class TestFastHist(unittest.TestCase):
    def test_fast_hist(self):
        tm._skip_if_no_sklearn()
        from sklearn.datasets import load_digits
        from sklearn.cross_validation import train_test_split

        digits = load_digits(2)
        X = digits['data']
        y = digits['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)

        param = {'objective': 'binary:logistic',
                 'tree_method': 'hist',
                 'grow_policy': 'depthwise',
                 'max_depth': 3,
                 'eval_metric': 'auc'}
        res = {}
        xgb.train(param, dtrain, 10, [(dtrain, 'train'), (dtest, 'test')],
                  evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert self.non_decreasing(res['test']['auc'])

        param2 = {'objective': 'binary:logistic',
                  'tree_method': 'hist',
                  'grow_policy': 'lossguide',
                  'max_depth': 0,
                  'max_leaves': 8,
                  'eval_metric': 'auc'}
        res = {}
        xgb.train(param2, dtrain, 10, [(dtrain, 'train'), (dtest, 'test')],
                  evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert self.non_decreasing(res['test']['auc'])

        param3 = {'objective': 'binary:logistic',
                  'tree_method': 'hist',
                  'grow_policy': 'lossguide',
                  'max_depth': 0,
                  'max_leaves': 8,
                  'max_bin': 16,
                  'eval_metric': 'auc'}
        res = {}
        xgb.train(param3, dtrain, 10, [(dtrain, 'train'), (dtest, 'test')],
                  evals_result=res)
        assert self.non_decreasing(res['train']['auc'])

        # fail-safe test for dense data
        from sklearn.datasets import load_svmlight_file
        dpath = 'demo/data/'
        X2, y2 = load_svmlight_file(dpath + 'agaricus.txt.train')
        X2 = X2.toarray()
        dtrain2 = xgb.DMatrix(X2, label=y2)

        param = {'objective': 'binary:logistic',
                 'tree_method': 'hist',
                 'grow_policy': 'depthwise',
                 'max_depth': 2,
                 'eval_metric': 'auc'}
        res = {}
        xgb.train(param, dtrain2, 10, [(dtrain2, 'train')], evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert res['train']['auc'][0] >= 0.85

        for j in range(X2.shape[1]):
            for i in np.random.choice(X2.shape[0], size=10, replace=False):
                X2[i, j] = 2

        dtrain3 = xgb.DMatrix(X2, label=y2)
        res = {}
        xgb.train(param, dtrain3, 10, [(dtrain3, 'train')], evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert res['train']['auc'][0] >= 0.85

        for j in range(X2.shape[1]):
            for i in np.random.choice(X2.shape[0], size=10, replace=False):
                X2[i, j] = 3

        dtrain4 = xgb.DMatrix(X2, label=y2)
        res = {}
        xgb.train(param, dtrain4, 10, [(dtrain4, 'train')], evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert res['train']['auc'][0] >= 0.85

        # fail-safe test for max_bin=2
        param = {'objective': 'binary:logistic',
                 'tree_method': 'hist',
                 'grow_policy': 'depthwise',
                 'max_depth': 2,
                 'eval_metric': 'auc',
                 'max_bin': 2}
        res = {}
        xgb.train(param, dtrain2, 10, [(dtrain2, 'train')], evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert res['train']['auc'][0] >= 0.85

    def non_decreasing(self, L):
        return all(x <= y for x, y in zip(L, L[1:]))
