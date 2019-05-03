import numpy as np
from scipy.sparse import csr_matrix
import xgboost

def test_ranking_with_unweighted_data():
    Xrow = np.array([1, 2, 6, 8, 11, 14, 16, 17])
    Xcol = np.array([0, 0, 1, 1,  2,  2,  3,  3])
    X = csr_matrix((np.ones(shape=8), (Xrow, Xcol)), shape=(20, 4))
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 1.0, 0.0,
                  0.0, 1.0, 0.0, 0.0, 1.0,
                  0.0, 1.0, 1.0, 0.0, 0.0])

    group = np.array([5, 5, 5, 5], dtype=np.uint)
    dtrain = xgboost.DMatrix(X, label=y)
    dtrain.set_group(group)

    params = {'eta': 1, 'tree_method': 'exact',
              'objective': 'rank:pairwise', 'eval_metric': ['auc', 'aucpr'],
              'max_depth': 1}
    evals_result = {}
    bst = xgboost.train(params, dtrain, 10, evals=[(dtrain, 'train')],
                        evals_result=evals_result)
    auc_rec = evals_result['train']['auc']
    assert all(p <= q for p, q in zip(auc_rec, auc_rec[1:]))
    auc_rec = evals_result['train']['aucpr']
    assert all(p <= q for p, q in zip(auc_rec, auc_rec[1:]))
