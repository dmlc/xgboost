import numpy as np
from scipy.sparse import csr_matrix
import xgboost

def test_ranking_with_weighted_data():
    Xrow = np.array([1, 2, 6, 8, 11, 14, 16, 17])
    Xcol = np.array([0, 0, 1, 1,  2,  2,  3,  3])
    X = csr_matrix((np.ones(shape=8), (Xrow, Xcol)), shape=(20, 4))
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 1.0, 0.0,
                  0.0, 1.0, 0.0, 0.0, 1.0,
                  0.0, 1.0, 1.0, 0.0, 0.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    group = np.array([5, 5, 5, 5], dtype=np.uint)
    dtrain = xgboost.DMatrix(X, label=y, weight=weights)
    dtrain.set_group(group)

    params = {'eta': 1, 'tree_method': 'exact',
              'objective': 'rank:pairwise', 'eval_metric': 'auc',
              'max_depth': 1}
    evals_result = {}
    bst = xgboost.train(params, dtrain, 10, evals=[(dtrain, 'train')],
                        evals_result=evals_result)
    auc_rec = evals_result['train']['auc']
    assert all(p <= q for p, q in zip(auc_rec, auc_rec[1:]))

    for i in range(1, 11):
        pred = bst.predict(dtrain, ntree_limit=i)
        # is_sorted[i]: is i-th group correctly sorted by the ranking predictor?
        is_sorted = []
        for k in range(0, 20, 5):
            ind = np.argsort(-pred[k:k+5])
            z = y[ind+k]
            is_sorted.append(all(i >= j for i, j in zip(z, z[1:])))
        # Since we give weights 1, 2, 3, 4 to the four query groups,
        # the ranking predictor will first try to correctly sort the last query group
        # before correctly sorting other groups.
        assert all(p <= q for p, q in zip(is_sorted, is_sorted[1:]))
