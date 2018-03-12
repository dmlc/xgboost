from __future__ import print_function

import itertools as it
import numpy as np
import sys
import os
import glob
import testing as tm
import unittest
import xgboost as xgb
try:
    from sklearn import metrics, datasets
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import scale
except ImportError:
    None

rng = np.random.RandomState(199)

num_rounds = 1000


def is_float(s):
    try:
        float(s)
        return 1
    except ValueError:
        return 0


def xgb_get_weights(bst):
    return np.array([float(s) for s in bst.get_dump()[0].split() if is_float(s)])


def check_ElasticNet(X, y, pred, tol, reg_alpha, reg_lambda, weights):
    enet = ElasticNet(alpha = reg_alpha + reg_lambda,
                      l1_ratio = reg_alpha/(reg_alpha + reg_lambda))
    enet.fit(X, y)
    enet_pred = enet.predict(X)
    assert np.isclose(weights, enet.coef_, rtol=tol, atol=tol).all()
    assert np.isclose(enet_pred, pred, rtol=tol, atol=tol).all()


def train_diabetes(param_in):
    data = datasets.load_diabetes()
    X = scale(data.data)
    dtrain = xgb.DMatrix(X, label=data.target)
    param = {}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    check_ElasticNet(X, data.target, xgb_pred, 1e-2,
                     param['alpha'], param['lambda'],
                     xgb_get_weights(bst)[1:])


def train_breast_cancer(param_in):
    data = datasets.load_breast_cancer()
    X = scale(data.data)
    dtrain = xgb.DMatrix(X, label=data.target)
    param = {'objective': 'binary:logistic'}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    xgb_score = metrics.accuracy_score(data.target, np.round(xgb_pred))
    assert xgb_score >= 0.8


def train_classification(param_in):
    X, y = datasets.make_classification(random_state=rng)
    X = scale(X)
    dtrain = xgb.DMatrix(X, label=y)
    param = {'objective': 'binary:logistic'}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    xgb_score = metrics.accuracy_score(y, np.round(xgb_pred))
    assert xgb_score >= 0.8


def train_classification_multi(param_in):
    num_class = 3
    X, y = datasets.make_classification(n_samples=100, random_state=rng,
                                        n_classes=num_class, n_informative=4,
                                        n_features=4, n_redundant=0)
    X = scale(X)
    dtrain = xgb.DMatrix(X, label=y)
    param = {'objective': 'multi:softmax', 'num_class': num_class}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    xgb_score = metrics.accuracy_score(y, np.round(xgb_pred))
    assert xgb_score >= 0.50


def train_boston(param_in):
    data = datasets.load_boston()
    X = scale(data.data)
    dtrain = xgb.DMatrix(X, label=data.target)
    param = {}
    param.update(param_in)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    check_ElasticNet(X, data.target, xgb_pred, 1e-2,
                     param['alpha'], param['lambda'],
                     xgb_get_weights(bst)[1:])


def train_external_mem(param_in):
    data = datasets.load_boston()
    X = scale(data.data)
    y = data.target
    param = {}
    param.update(param_in)
    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred = bst.predict(dtrain)
    np.savetxt('tmptmp_1234.csv', np.hstack((y.reshape(len(y),1), X)),
               delimiter=',', fmt='%10.9f')
    dtrain = xgb.DMatrix('tmptmp_1234.csv?format=csv&label_column=0#tmptmp_')
    bst = xgb.train(param, dtrain, num_rounds)
    xgb_pred_ext = bst.predict(dtrain)
    assert np.abs(xgb_pred_ext - xgb_pred).max() < 1e-3
    dtrain.__del__()
    bst.__del__()
    for f in glob.glob("tmptmp_*"):
        os.remove(f)


# Enumerates all permutations of variable parameters
def assert_updater_accuracy(linear_updater, variable_param):
    param = {'booster': 'gblinear', 'updater': linear_updater, 'eta': 1.,
             'top_k': 10, 'tolerance': 1e-5, 'ntread': 3}
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
        train_external_mem(param_tmp)


class TestLinear(unittest.TestCase):
    def test_coordinate(self):
        tm._skip_if_no_sklearn()
        variable_param = {'alpha': [.005, .1], 'lambda': [.005],
                          'feature_selector': ['cyclic', 'shuffle', 'greedy', 'thrifty']}
        assert_updater_accuracy('coord_descent', variable_param)

    def test_shotgun(self):
        tm._skip_if_no_sklearn()
        variable_param = {'alpha': [.005, .1], 'lambda': [.005, .1]}
        assert_updater_accuracy('shotgun', variable_param)
