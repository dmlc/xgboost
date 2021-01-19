'''Tests for running inplace prediction.'''
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse

import xgboost as xgb


def run_threaded_predict(X, rows, predict_func):
    results = []
    per_thread = 20
    with ThreadPoolExecutor(max_workers=10) as e:
        for i in range(0, rows, int(rows / per_thread)):
            if hasattr(X, 'iloc'):
                predictor = X.iloc[i:i+per_thread, :]
            else:
                predictor = X[i:i+per_thread, ...]
            f = e.submit(predict_func, predictor)
            results.append(f)

    for f in results:
        assert f.result()


def run_predict_leaf(predictor):
    rows = 100
    cols = 4
    classes = 5
    num_parallel_tree = 4
    num_boost_round = 10
    rng = np.random.RandomState(1994)
    X = rng.randn(rows, cols)
    y = rng.randint(low=0, high=classes, size=rows)
    m = xgb.DMatrix(X, y)
    booster = xgb.train(
        {
            "num_parallel_tree": num_parallel_tree,
            "num_class": classes,
            "predictor": predictor,
            "tree_method": "hist",
        },
        m,
        num_boost_round=num_boost_round,
    )

    empty = xgb.DMatrix(np.ones(shape=(0, cols)))
    empty_leaf = booster.predict(empty, pred_leaf=True)
    assert empty_leaf.shape[0] == 0

    leaf = booster.predict(m, pred_leaf=True)
    assert leaf.shape[0] == rows
    assert leaf.shape[1] == classes * num_parallel_tree * num_boost_round

    for i in range(rows):
        row = leaf[i, ...]
        for j in range(num_boost_round):
            start = classes * num_parallel_tree * j
            end = classes * num_parallel_tree * (j + 1)
            layer = row[start: end]
            for c in range(classes):
                tree_group = layer[c * num_parallel_tree: (c + 1) * num_parallel_tree]
                assert tree_group.shape[0] == num_parallel_tree
                # no subsampling so tree in same forest should output same
                # leaf.
                assert np.all(tree_group == tree_group[0])

    ntree_limit = 2
    sliced = booster.predict(
        m, pred_leaf=True, ntree_limit=num_parallel_tree * ntree_limit
    )
    first = sliced[0, ...]

    assert first.shape[0] == classes * num_parallel_tree * ntree_limit
    return leaf


def test_predict_leaf():
    run_predict_leaf('cpu_predictor')


class TestInplacePredict:
    '''Tests for running inplace prediction'''
    def test_predict(self):
        rows = 1000
        cols = 10

        np.random.seed(1994)

        X = np.random.randn(rows, cols)
        y = np.random.randn(rows)
        dtrain = xgb.DMatrix(X, y)

        booster = xgb.train({'tree_method': 'hist'},
                            dtrain, num_boost_round=10)

        test = xgb.DMatrix(X[:10, ...])
        predt_from_array = booster.inplace_predict(X[:10, ...])
        predt_from_dmatrix = booster.predict(test)

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        predt_from_array = booster.inplace_predict(X[:10, ...], iteration_range=(0, 4))
        predt_from_dmatrix = booster.predict(test, ntree_limit=4)

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        def predict_dense(x):
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, rows, predict_dense)

        def predict_csr(x):
            inplace_predt = booster.inplace_predict(sparse.csr_matrix(x))
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, rows, predict_csr)
