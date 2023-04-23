import os
import subprocess
import tempfile

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

pytestmark = tm.timeout(10)


class TestOMP:
    def test_omp(self):
        dtrain, dtest = tm.load_agaricus(__file__)

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

        n_trials = 10
        param['nthread'] = 1
        auc_1, pred_1 = consist_test('approx_thread_1', n_trials)

        param['nthread'] = 2
        auc_2, pred_2 = consist_test('approx_thread_2', n_trials)

        param['nthread'] = 3
        auc_3, pred_3 = consist_test('approx_thread_3', n_trials)

        assert auc_1 == auc_2 == auc_3
        assert np.array_equal(auc_1, auc_2)
        assert np.array_equal(auc_1, auc_3)

        print('test hist ...')
        param['tree_method'] = 'hist'

        param['nthread'] = 1
        auc_1, pred_1 = consist_test('hist_thread_1', n_trials)

        param['nthread'] = 2
        auc_2, pred_2 = consist_test('hist_thread_2', n_trials)

        param['nthread'] = 3
        auc_3, pred_3 = consist_test('hist_thread_3', n_trials)

        assert auc_1 == auc_2 == auc_3
        assert np.array_equal(auc_1, auc_2)
        assert np.array_equal(auc_1, auc_3)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_with_omp_thread_limit(self):
        args = [
            "python", os.path.join(
                os.path.dirname(tm.normpath(__file__)), "with_omp_limit.py"
            )
        ]
        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in (1, 2, 16):
                path = os.path.join(tmpdir, str(i))
                with open(path, "w") as fd:
                    fd.write("\n")
                cp = args.copy()
                cp.append(path)

                env = os.environ.copy()
                env["OMP_THREAD_LIMIT"] = str(i)

                status = subprocess.call(cp, env=env)
                assert status == 0

                with open(path, "r") as fd:
                    results.append(float(fd.read()))

        for auc in results:
            np.testing.assert_allclose(auc, results[0])
