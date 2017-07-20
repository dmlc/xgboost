from __future__ import print_function
#pylint: skip-file
import sys
sys.path.append("../../tests/python")
import xgboost as xgb
import testing as tm
import numpy as np
import unittest
from sklearn.datasets import make_classification
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs) ; sys.stderr.flush()
    print(*args, file=sys.stdout, **kwargs) ; sys.stdout.flush()

eprint("Testing Big Data (this may take a while)")

rng = np.random.RandomState(1994)

# "realistic" size based upon http://stat-computing.org/dataexpo/2009/ , which has been processed to one-hot encode categoricalsxsy
cols = 31
# reduced to fit onto 1 gpu but still be large
rows3 = 5000 # small
rows2 = 4360032 # medium
rows1 = 42360032 # large
#rows1 = 152360032 # can do this for multi-gpu test (very large)
rowslist = [rows1, rows2, rows3]


class TestGPU(unittest.TestCase):
    def test_large(self):
        eprint("Starting test for large data")
        tm._skip_if_no_sklearn()

        for rows in rowslist:
            
            eprint("Creating train data rows=%d cols=%d" % (rows,cols))
            np.random.seed(7)
            X = np.random.rand(rows, cols)
            y = np.random.rand(rows)
            eprint("Starting DMatrix(X,y)")
            ag_dtrain = xgb.DMatrix(X,y,nthread=0)

            max_depth=6
            max_bin=1024
                
            # regression test --- hist must be same as exact on all-categorial data
            ag_param = {'max_depth': max_depth,
                        'tree_method': 'exact',
                        'nthread': 0,
                        'eta': 1,
                        'silent': 0,
                        'objective': 'binary:logistic',
                        'eval_metric': 'auc'}
            ag_paramb = {'max_depth': max_depth,
                        'tree_method': 'hist',
                        'nthread': 0,
                        'eta': 1,
                        'silent': 0,
                        'objective': 'binary:logistic',
                        'eval_metric': 'auc'}
            ag_param2 = {'max_depth': max_depth,
                         'tree_method': 'gpu_hist',
                         'nthread': 0,
                         'eta': 1,
                         'silent': 0,
                         'n_gpus': 1,
                         'objective': 'binary:logistic',
                         'max_bin': max_bin,
                         'eval_metric': 'auc'}
            ag_param3 = {'max_depth': max_depth,
                         'tree_method': 'gpu_hist',
                         'nthread': 0,
                         'eta': 1,
                         'silent': 0,
                         'n_gpus': -1,
                         'objective': 'binary:logistic',
                         'max_bin': max_bin,
                         'eval_metric': 'auc'}
            ag_res = {}
            ag_resb = {}
            ag_res2 = {}
            ag_res3 = {}

            num_rounds = 1
            
            eprint("normal updater")
            xgb.train(ag_param, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                      evals_result=ag_res)
            eprint("hist updater")
            xgb.train(ag_paramb, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                      evals_result=ag_resb)
            eprint("gpu_hist updater 1 gpu")
            xgb.train(ag_param2, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                      evals_result=ag_res2)
            eprint("gpu_hist updater all gpus")
            xgb.train(ag_param3, ag_dtrain, num_rounds, [(ag_dtrain, 'train'), (ag_dtest, 'test')],
                      evals_result=ag_res3)

            assert np.fabs(ag_res['train']['auc'][0] - ag_resb['train']['auc'][0])<0.001
            assert np.fabs(ag_res['train']['auc'][0] - ag_res2['train']['auc'][0])<0.001
            assert np.fabs(ag_res['train']['auc'][0] - ag_res3['train']['auc'][0])<0.001

    
    
