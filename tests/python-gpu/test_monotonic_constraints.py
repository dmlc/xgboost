import sys
import numpy as np

import pytest

import xgboost as xgb
sys.path.append("tests/python")
import testing as tm
import test_monotone_constraints as tmc

rng = np.random.RandomState(1994)


def non_decreasing(L):
    return all((x - y) < 0.001 for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all((y - x) < 0.001 for x, y in zip(L, L[1:]))


@pytest.mark.skipif(**tm.no_sklearn())
@pytest.mark.parametrize('constraint', [1, -1],
                         ids=['nondecreasing_constraint', 'nonincreasing_constraint'])
def test_gpu_hist_basic(constraint):
    from sklearn.datasets import make_regression
    n = 1000
    X, y = make_regression(n, random_state=rng, n_features=1, n_informative=1)
    dtrain = xgb.DMatrix(X, y)
    param = {}
    param['tree_method'] = 'gpu_hist'
    param['monotone_constraints'] = "(" + str(constraint) + ")"
    bst = xgb.train(param, dtrain)
    dpredict = xgb.DMatrix(X[X[:, 0].argsort()])
    pred = bst.predict(dpredict)

    if constraint > 0:
        assert non_decreasing(pred)
    elif constraint < 0:
        assert non_increasing(pred)


@pytest.mark.parametrize('grow_policy', ['depthwise', 'lossguide'])
def test_gpu_hist(grow_policy):
    params = {
        'tree_method': 'gpu_hist',
        'grow_policy': grow_policy,
        'monotone_constraints': '(1, -1)'
    }
    model = xgb.train(params, tmc.training_dset)
    tmc.is_correctly_constrained(model)
