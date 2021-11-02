from multiprocessing import Process, Queue
import os

import xgboost as xgb
import numpy as np
import pytest

import testing as tm


class TestOMP:
    def test_omp(self):
        dpath = 'demo/data/'
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

        param = {'booster': 'gbtree',
                 'objective': 'binary:logistic',
                 'grow_policy': 'depthwise',
                 'tree_method': 'hist',
                 'eval_metric': 'error',
                 'max_depth': 5,
                 'min_child_weight': 0}

        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 5

        def run_trial():
            res = {}
            bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=res)
            metrics = [res['train']['error'][-1], res['eval']['error'][-1]]
            preds = bst.predict(dtest)
            return metrics, preds

        def consist_test(title, n):
            auc, pred = run_trial()
            for i in range(n-1):
                auc2, pred2 = run_trial()
                try:
                    assert auc == auc2
                    assert np.array_equal(pred, pred2)
                except Exception as e:
                    print('-------test %s failed, num_trial: %d-------' % (title, i))
                    raise e
                auc, pred = auc2, pred2
            return auc, pred

        print('test approx ...')
        param['tree_method'] = 'approx'

        param['nthread'] = 1
        auc_1, pred_1 = consist_test('approx_thread_1', 100)

        param['nthread'] = 2
        auc_2, pred_2 = consist_test('approx_thread_2', 100)

        param['nthread'] = 3
        auc_3, pred_3 = consist_test('approx_thread_3', 100)

        assert auc_1 == auc_2 == auc_3
        assert np.array_equal(auc_1, auc_2)
        assert np.array_equal(auc_1, auc_3)

        print('test hist ...')
        param['tree_method'] = 'hist'

        param['nthread'] = 1
        auc_1, pred_1 = consist_test('hist_thread_1', 100)

        param['nthread'] = 2
        auc_2, pred_2 = consist_test('hist_thread_2', 100)

        param['nthread'] = 3
        auc_3, pred_3 = consist_test('hist_thread_3', 100)

        assert auc_1 == auc_2 == auc_3
        assert np.array_equal(auc_1, auc_2)
        assert np.array_equal(auc_1, auc_3)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_with_omp_thread_limit(self):
        def test_fn(q: Queue, limit: int):
            from sklearn.datasets import load_iris
            from sklearn.metrics import roc_auc_score
            os.environ["OMP_THREAD_LIMIT"] = str(limit)
            X, y = load_iris(return_X_y=True)
            Xy = xgb.DMatrix(X, y, nthread=16)
            booster = xgb.train({"num_class": 3, "objective": "multi:softprob"}, Xy, num_boost_round=8)
            score = booster.predict(Xy)
            print(y.shape, score.shape)
            auc = roc_auc_score(y, score, average="weighted", multi_class="ovr")
            q.put(auc)

        queue = Queue()
        processes = []
        for i in (1, 2, 16):
            p = Process(target=test_fn, args=(queue, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            score = queue.get()
            print("score:", score)
