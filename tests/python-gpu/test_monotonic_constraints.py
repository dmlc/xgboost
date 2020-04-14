import sys
import numpy as np

import unittest
import pytest

import xgboost as xgb
sys.path.append("tests/python")
import testing as tm

rng = np.random.RandomState(1994)


def non_decreasing(L):
    return all((x - y) < 0.001 for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all((y - x) < 0.001 for x, y in zip(L, L[1:]))


def assert_constraint(constraint, tree_method):
    from sklearn.datasets import make_regression
    n = 1000
    X, y = make_regression(n, random_state=rng, n_features=1, n_informative=1)
    dtrain = xgb.DMatrix(X, y)
    param = {}
    param['tree_method'] = tree_method
    param['monotone_constraints'] = "(" + str(constraint) + ")"
    bst = xgb.train(param, dtrain)
    dpredict = xgb.DMatrix(X[X[:, 0].argsort()])
    pred = bst.predict(dpredict)
    if constraint > 0:
        assert non_decreasing(pred)
    elif constraint < 0:
        assert non_increasing(pred)


class TestMonotonicConstraints(unittest.TestCase):
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_exact(self):
        assert_constraint(1, 'exact')
        assert_constraint(-1, 'exact')

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_gpu_hist(self):
        assert_constraint(1, 'gpu_hist')
        assert_constraint(-1, 'gpu_hist')
