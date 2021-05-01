'''Tests for running inplace prediction.'''
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse
import pytest
import pandas as pd

import testing as tm
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


def verify_leaf_output(leaf: np.ndarray, num_parallel_tree: int):
    for i in range(leaf.shape[0]):     # n_samples
        for j in range(leaf.shape[1]):  # n_rounds
            for k in range(leaf.shape[2]):    # n_classes
                tree_group = leaf[i, j, k, :]
                assert tree_group.shape[0] == num_parallel_tree
                # No sampling, all trees within forest are the same
                assert np.all(tree_group == tree_group[0])


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

    leaf = booster.predict(m, pred_leaf=True, strict_shape=True)
    assert leaf.shape[0] == rows
    assert leaf.shape[1] == num_boost_round
    assert leaf.shape[2] == classes
    assert leaf.shape[3] == num_parallel_tree

    verify_leaf_output(leaf, num_parallel_tree)

    ntree_limit = 2
    sliced = booster.predict(
        m, pred_leaf=True, ntree_limit=num_parallel_tree * ntree_limit, strict_shape=True
    )
    first = sliced[0, ...]

    assert np.prod(first.shape) == classes * num_parallel_tree * ntree_limit

    # When there's only 1 tree, the output is a 1 dim vector
    booster = xgb.train({"tree_method": "hist"}, num_boost_round=1, dtrain=m)
    assert booster.predict(m, pred_leaf=True).shape == (rows, )

    return leaf


def test_predict_leaf():
    run_predict_leaf('cpu_predictor')


def test_predict_shape():
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    reg = xgb.XGBRegressor(n_estimators=1)
    reg.fit(X, y)
    predt = reg.get_booster().predict(xgb.DMatrix(X), strict_shape=True)
    assert len(predt.shape) == 2
    assert predt.shape[0] == X.shape[0]
    assert predt.shape[1] == 1

    contrib = reg.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True, strict_shape=True
    )
    assert len(contrib.shape) == 3
    assert contrib.shape[1] == 1

    contrib = reg.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True, approx_contribs=True
    )
    assert len(contrib.shape) == 2
    assert contrib.shape[1] == X.shape[1] + 1

    interaction = reg.get_booster().predict(
        xgb.DMatrix(X), pred_interactions=True, approx_contribs=True
    )
    assert len(interaction.shape) == 3
    assert interaction.shape[1] == X.shape[1] + 1
    assert interaction.shape[2] == X.shape[1] + 1

    interaction = reg.get_booster().predict(
        xgb.DMatrix(X), pred_interactions=True, approx_contribs=True, strict_shape=True
    )
    assert len(interaction.shape) == 4
    assert interaction.shape[1] == 1
    assert interaction.shape[2] == X.shape[1] + 1
    assert interaction.shape[3] == X.shape[1] + 1


class TestInplacePredict:
    '''Tests for running inplace prediction'''
    @classmethod
    def setup_class(cls):
        cls.rows = 1000
        cls.cols = 10

        cls.missing = 11            # set to integer for testing

        cls.rng = np.random.RandomState(1994)

        cls.X = cls.rng.randn(cls.rows, cls.cols)
        missing_idx = [i for i in range(0, cls.cols, 4)]
        cls.X[:, missing_idx] = cls.missing  # set to be missing

        cls.y = cls.rng.randn(cls.rows)

        dtrain = xgb.DMatrix(cls.X, cls.y)
        cls.test = xgb.DMatrix(cls.X[:10, ...], missing=cls.missing)

        cls.booster = xgb.train({'tree_method': 'hist'}, dtrain, num_boost_round=10)

    def test_predict(self):
        booster = self.booster
        X = self.X
        test = self.test

        predt_from_array = booster.inplace_predict(X[:10, ...], missing=self.missing)
        predt_from_dmatrix = booster.predict(test)

        X_obj = X.copy().astype(object)

        assert X_obj.dtype.hasobject is True
        assert X.dtype.hasobject is False
        np.testing.assert_allclose(
            booster.inplace_predict(X_obj), booster.inplace_predict(X)
        )

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        predt_from_array = booster.inplace_predict(
            X[:10, ...], iteration_range=(0, 4), missing=self.missing
        )
        predt_from_dmatrix = booster.predict(test, ntree_limit=4)

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        def predict_dense(x):
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, self.rows, predict_dense)

        def predict_csr(x):
            inplace_predt = booster.inplace_predict(sparse.csr_matrix(x))
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, self.rows, predict_csr)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_predict_pd(self):
        X = self.X
        # construct it in column major style
        df = pd.DataFrame({str(i): X[:, i] for i in range(X.shape[1])})
        booster = self.booster
        df_predt = booster.inplace_predict(df)
        arr_predt = booster.inplace_predict(X)
        dmat_predt = booster.predict(xgb.DMatrix(X))

        X = df.values
        X = np.asfortranarray(X)
        fort_predt = booster.inplace_predict(X)

        np.testing.assert_allclose(dmat_predt, arr_predt)
        np.testing.assert_allclose(df_predt, arr_predt)
        np.testing.assert_allclose(fort_predt, arr_predt)

    def test_base_margin(self):
        booster = self.booster

        base_margin = self.rng.randn(self.rows)
        from_inplace = booster.inplace_predict(data=self.X, base_margin=base_margin)

        dtrain = xgb.DMatrix(self.X, self.y, base_margin=base_margin)
        from_dmatrix = booster.predict(dtrain)
        np.testing.assert_allclose(from_dmatrix, from_inplace)
