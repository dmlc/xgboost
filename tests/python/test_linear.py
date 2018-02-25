from __future__ import print_function

import itertools as it
import numpy as np
import sys
import testing as tm
import unittest
import xgboost as xgb

rng = np.random.RandomState(199)

num_rounds = 1000


def is_float(s):
    try:
        float(s)
        return 1
    except ValueError:
        return 0


def xgb_get_weights(bst):
    return [float(s) for s in bst.get_dump()[0].split() if is_float(s)]


# Check gradient/subgradient = 0
def check_least_squares_solution(X, y, pred, tol, reg_alpha, reg_lambda, weights):
    reg_alpha = reg_alpha * len(y)
    reg_lambda = reg_lambda * len(y)
    r = np.subtract(y, pred)
    g = X.T.dot(r)
    g = np.subtract(g, np.multiply(reg_lambda, weights))
    for i in range(0, len(weights)):
        if weights[i] == 0.0:
            assert abs(g[i]) <= reg_alpha
        else:
            assert np.isclose(g[i], np.sign(weights[i]) * reg_alpha, rtol=tol, atol=tol)


def train_diabetes(param_in):
    from sklearn import datasets
    data = datasets.load_diabetes()
    dtrain = xgb.DMatrix(data.data, label=data.target)
    param = {}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    check_least_squares_solution(data.data, data.target, xgb_pred, 1e-2, param['alpha'], param['lambda'],
                                 xgb_get_weights(bst)[1:])


def train_breast_cancer(param_in):
    from sklearn import metrics, datasets
    data = datasets.load_breast_cancer()
    dtrain = xgb.DMatrix(data.data, label=data.target)
    param = {'objective': 'binary:logistic'}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    xgb_score = metrics.accuracy_score(data.target, np.round(xgb_pred))
    assert xgb_score >= 0.8


def train_classification(param_in):
    from sklearn import metrics, datasets
    X, y = datasets.make_classification(random_state=rng,
                                        scale=100)  # Scale is necessary otherwise regularisation parameters will force all coefficients to 0
    dtrain = xgb.DMatrix(X, label=y)
    param = {'objective': 'binary:logistic'}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    xgb_score = metrics.accuracy_score(y, np.round(xgb_pred))
    assert xgb_score >= 0.8


def train_classification_multi(param_in):
    from sklearn import metrics, datasets
    num_class = 3
    X, y = datasets.make_classification(n_samples=10, random_state=rng, scale=100, n_classes=num_class, n_informative=4,
                                        n_features=4, n_redundant=0)
    dtrain = xgb.DMatrix(X, label=y)
    param = {'objective': 'multi:softmax', 'num_class': num_class}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    xgb_score = metrics.accuracy_score(y, np.round(xgb_pred))
    assert xgb_score >= 0.50


def train_boston(param_in):
    from sklearn import datasets
    data = datasets.load_boston()
    dtrain = xgb.DMatrix(data.data, label=data.target)
    param = {}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    check_least_squares_solution(data.data, data.target, xgb_pred, 1e-2, param['alpha'], param['lambda'],
                                 xgb_get_weights(bst)[1:])


# Enumerates all permutations of variable parameters
def assert_updater_accuracy(linear_updater, variable_param):
    param = {'booster': 'gblinear', 'updater': linear_updater, 'eta': 1., 'tolerance': 1e-8}
    names = sorted(variable_param)
    combinations = it.product(*(variable_param[Name] for Name in names))

    for set in combinations:
        param_tmp = param.copy()
        for i, name in enumerate(names):
            param_tmp[name] = set[i]

        print(param_tmp, file=sys.stderr)
        train_boston(param_tmp)
        train_diabetes(param_tmp)
        train_classification(param_tmp)
        train_classification_multi(param_tmp)
        train_breast_cancer(param_tmp)


class TestLinear(unittest.TestCase):
    def test_coordinate(self):
        tm._skip_if_no_sklearn()
        variable_param = {'alpha': [1.0, 5.0], 'lambda': [1.0, 5.0],
                          'coordinate_selection': ['cyclic', 'random', 'greedy']}
        assert_updater_accuracy('coord_descent', variable_param)

    def test_shotgun(self):
        tm._skip_if_no_sklearn()
        variable_param = {'alpha': [1.0, 5.0], 'lambda': [1.0, 5.0]}
        assert_updater_accuracy('shotgun', variable_param)
