from __future__ import print_function
#pylint: skip-file
import sys
sys.path.append("../../tests/python")
import xgboost as xgb
import testing as tm
import numpy as np
import unittest

rng = np.random.RandomState(1994)

dpath = '../../demo/data/'
ag_dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
ag_dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print(*args, file=sys.stdout, **kwargs)
        

class TestGPU(unittest.TestCase):
    def test_grow_gpu(self):
        tm._skip_if_no_sklearn()
        from sklearn.datasets import load_digits
        try:
            from sklearn.model_selection import train_test_split
        except:
            from sklearn.cross_validation import train_test_split

        ag_param = {'max_depth': 2,
                    'tree_method': 'exact',
                    'nthread': 1,
                    'eta': 1,
                    'silent': 1,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc'}
        ag_param2 = {'max_depth': 2,
                     'tree_method': 'gpu_exact',
                     'eta': 1,
                     'silent': 1,
                     'objective': 'binary:logistic',
                     'eval_metric': 'auc'}
        ag_res = {}
        ag_res2 = {}

        num_rounds = 10
        xgb.train(ag_param, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                  evals_result=ag_res)
        xgb.train(ag_param2, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                  evals_result=ag_res2)
        assert ag_res['train']['auc'] == ag_res2['train']['auc']
        assert ag_res['test']['auc'] == ag_res2['test']['auc']

        digits = load_digits(2)
        X = digits['data']
        y = digits['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)

        param = {'objective': 'binary:logistic',
                 'tree_method': 'gpu_exact',
                 'max_depth': 3,
                 'eval_metric': 'auc'}
        res = {}
        xgb.train(param, dtrain, num_rounds, [(dtrain, 'train'), (dtest, 'test')],
                  evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert self.non_decreasing(res['test']['auc'])

        # fail-safe test for dense data
        from sklearn.datasets import load_svmlight_file
        X2, y2 = load_svmlight_file(dpath + 'agaricus.txt.train')
        X2 = X2.toarray()
        dtrain2 = xgb.DMatrix(X2, label=y2)

        param = {'objective': 'binary:logistic',
                 'tree_method': 'gpu_exact',
                 'max_depth': 2,
                 'eval_metric': 'auc'}
        res = {}
        xgb.train(param, dtrain2, num_rounds, [(dtrain2, 'train')], evals_result=res)

        assert self.non_decreasing(res['train']['auc'])
        assert res['train']['auc'][0] >= 0.85

        for j in range(X2.shape[1]):
            for i in rng.choice(X2.shape[0], size=num_rounds, replace=False):
                X2[i, j] = 2

        dtrain3 = xgb.DMatrix(X2, label=y2)
        res = {}

        xgb.train(param, dtrain3, num_rounds, [(dtrain3, 'train')], evals_result=res)

        assert self.non_decreasing(res['train']['auc'])
        assert res['train']['auc'][0] >= 0.85

        for j in range(X2.shape[1]):
            for i in np.random.choice(X2.shape[0], size=num_rounds, replace=False):
                X2[i, j] = 3

        dtrain4 = xgb.DMatrix(X2, label=y2)
        res = {}
        xgb.train(param, dtrain4, num_rounds, [(dtrain4, 'train')], evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        assert res['train']['auc'][0] >= 0.85

        
    def test_grow_gpu_hist(self):
        n_gpus=-1
        tm._skip_if_no_sklearn()
        from sklearn.datasets import load_digits
        try:
            from sklearn.model_selection import train_test_split
        except:
            from sklearn.cross_validation import train_test_split

        for max_depth in range(3,10): # TODO: Doesn't work with 2 for some tests
            #eprint("max_depth=%d" % (max_depth))
            
            for max_bin_i in range(3,11):
                max_bin = np.power(2,max_bin_i)
                #eprint("max_bin=%d" % (max_bin))

                
            
                # regression test --- hist must be same as exact on all-categorial data
                ag_param = {'max_depth': max_depth,
                            'tree_method': 'exact',
                            'nthread': 1,
                            'eta': 1,
                            'silent': 1,
                            'objective': 'binary:logistic',
                            'eval_metric': 'auc'}
                ag_param2 = {'max_depth': max_depth,
                             'tree_method': 'gpu_hist',
                             'eta': 1,
                             'silent': 1,
                             'n_gpus': 1,
                             'objective': 'binary:logistic',
                                 'max_bin': max_bin,
                             'eval_metric': 'auc'}
                ag_param3 = {'max_depth': max_depth,
                             'tree_method': 'gpu_hist',
                             'eta': 1,
                             'silent': 1,
                             'n_gpus': n_gpus,
                                 'objective': 'binary:logistic',
                                 'max_bin': max_bin,
                             'eval_metric': 'auc'}
                ag_res = {}
                ag_res2 = {}
                ag_res3 = {}

                num_rounds = 10
                #eprint("normal updater");
                xgb.train(ag_param, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                          evals_result=ag_res)
                #eprint("grow_gpu_hist updater 1 gpu");
                xgb.train(ag_param2, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                          evals_result=ag_res2)
                #eprint("grow_gpu_hist updater %d gpus" % (n_gpus));
                xgb.train(ag_param3, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                          evals_result=ag_res3)
                #        assert 1==0
                assert ag_res['train']['auc'] == ag_res2['train']['auc']
                assert ag_res['test']['auc'] == ag_res2['test']['auc']
                assert ag_res['test']['auc'] == ag_res3['test']['auc']

                ######################################################################
                digits = load_digits(2)
                X = digits['data']
                y = digits['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
                dtrain = xgb.DMatrix(X_train, y_train)
                dtest = xgb.DMatrix(X_test, y_test)

                param = {'objective': 'binary:logistic',
                         'tree_method': 'gpu_hist',
                         'max_depth': max_depth,
                         'n_gpus': 1,
                         'max_bin': max_bin,
                         'eval_metric': 'auc'}
                res = {}
                #eprint("digits: grow_gpu_hist updater 1 gpu");
                xgb.train(param, dtrain, num_rounds, [(dtrain, 'train'), (dtest, 'test')],
                          evals_result=res)
                assert self.non_decreasing(res['train']['auc'])
                #assert self.non_decreasing(res['test']['auc'])
                param2 = {'objective': 'binary:logistic',
                          'tree_method': 'gpu_hist',
                          'max_depth': max_depth,
                          'n_gpus': n_gpus,
                          'max_bin': max_bin,
                          'eval_metric': 'auc'}
                res2 = {}
                #eprint("digits: grow_gpu_hist updater %d gpus" % (n_gpus));
                xgb.train(param2, dtrain, num_rounds, [(dtrain, 'train'), (dtest, 'test')],
                          evals_result=res2)
                assert self.non_decreasing(res2['train']['auc'])
                #assert self.non_decreasing(res2['test']['auc'])
                assert res['train']['auc'] == res2['train']['auc']
                #assert res['test']['auc'] == res2['test']['auc']

                ######################################################################
                # fail-safe test for dense data
                from sklearn.datasets import load_svmlight_file
                X2, y2 = load_svmlight_file(dpath + 'agaricus.txt.train')
                X2 = X2.toarray()
                dtrain2 = xgb.DMatrix(X2, label=y2)

                param = {'objective': 'binary:logistic',
                         'tree_method': 'gpu_hist',
                         'max_depth': max_depth,
                         'n_gpus': n_gpus,
                         'max_bin': max_bin,
                         'eval_metric': 'auc'}
                res = {}
                xgb.train(param, dtrain2, num_rounds, [(dtrain2, 'train')], evals_result=res)

                assert self.non_decreasing(res['train']['auc'])
                if max_bin>32:
                    assert res['train']['auc'][0] >= 0.85

                for j in range(X2.shape[1]):
                    for i in rng.choice(X2.shape[0], size=num_rounds, replace=False):
                        X2[i, j] = 2

                dtrain3 = xgb.DMatrix(X2, label=y2)
                res = {}

                xgb.train(param, dtrain3, num_rounds, [(dtrain3, 'train')], evals_result=res)

                assert self.non_decreasing(res['train']['auc'])
                if max_bin>32:
                    assert res['train']['auc'][0] >= 0.85

                for j in range(X2.shape[1]):
                    for i in np.random.choice(X2.shape[0], size=num_rounds, replace=False):
                        X2[i, j] = 3

                dtrain4 = xgb.DMatrix(X2, label=y2)
                res = {}
                xgb.train(param, dtrain4, num_rounds, [(dtrain4, 'train')], evals_result=res)
                assert self.non_decreasing(res['train']['auc'])
                if max_bin>32:
                    assert res['train']['auc'][0] >= 0.85

                ######################################################################
                # fail-safe test for max_bin
                param = {'objective': 'binary:logistic',
                         'tree_method': 'gpu_hist',
                         'max_depth': max_depth,
                         'n_gpus': n_gpus,
                         'eval_metric': 'auc',
                         'max_bin': max_bin}
                res = {}
                xgb.train(param, dtrain2, num_rounds, [(dtrain2, 'train')], evals_result=res)
                assert self.non_decreasing(res['train']['auc'])
                if max_bin>32:
                    assert res['train']['auc'][0] >= 0.85
                ######################################################################
                # subsampling
                param = {'objective': 'binary:logistic',
                         'tree_method': 'gpu_hist',
                         'max_depth': max_depth,
                         'n_gpus': n_gpus,
                         'eval_metric': 'auc',
                         'colsample_bytree': 0.5,
                         'colsample_bylevel': 0.5,
                         'subsample': 0.5,
                         'max_bin': max_bin}
                res = {}
                xgb.train(param, dtrain2, num_rounds, [(dtrain2, 'train')], evals_result=res)
                assert self.non_decreasing(res['train']['auc'])
                if max_bin>32:
                    assert res['train']['auc'][0] >= 0.85
        ######################################################################
        # fail-safe test for max_bin=2
        param = {'objective': 'binary:logistic',
                 'tree_method': 'gpu_hist',
                 'max_depth': 2,
                 'n_gpus': n_gpus,
                 'eval_metric': 'auc',
                 'max_bin': 2}
        res = {}
        xgb.train(param, dtrain2, num_rounds, [(dtrain2, 'train')], evals_result=res)
        assert self.non_decreasing(res['train']['auc'])
        if max_bin>32:
            assert res['train']['auc'][0] >= 0.85
        
        
    def non_decreasing(self, L):
            return all((x - y) < 0.001 for x, y in zip(L, L[1:]))
