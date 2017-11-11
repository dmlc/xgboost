from __future__ import print_function

import sys

sys.path.append("../../tests/python")
import xgboost as xgb
import numpy as np
import unittest
from nose.plugins.attrib import attr
from sklearn.datasets import load_digits, load_boston, load_breast_cancer, make_regression
import itertools as it

rng = np.random.RandomState(1994)


def non_increasing(L, tolerance):
    return all((y - x) < tolerance for x, y in zip(L, L[1:]))


# Check result is always decreasing and final accuracy is within tolerance
def assert_accuracy(res, tree_method, comparison_tree_method, tolerance, param):
    assert non_increasing(res[tree_method], tolerance)
    assert np.allclose(res[tree_method][-1], res[comparison_tree_method][-1], 1e-3, 1e-2)


def train_boston(param_in, comparison_tree_method):
    data = load_boston()
    dtrain = xgb.DMatrix(data.data, label=data.target)
    param = {}
    param.update(param_in)
    param['max_depth'] = 2
    res_tmp = {}
    res = {}
    num_rounds = 10
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[param['tree_method']] = res_tmp['train']['rmse']
    param["tree_method"] = comparison_tree_method
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[comparison_tree_method] = res_tmp['train']['rmse']

    return res


def train_digits(param_in, comparison_tree_method):
    data = load_digits()
    dtrain = xgb.DMatrix(data.data, label=data.target)
    param = {}
    param['objective'] = 'multi:softmax'
    param['num_class'] = 10
    param.update(param_in)
    res_tmp = {}
    res = {}
    num_rounds = 10
    xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[param['tree_method']] = res_tmp['train']['merror']
    param["tree_method"] = comparison_tree_method
    xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[comparison_tree_method] = res_tmp['train']['merror']
    return res


def train_cancer(param_in, comparison_tree_method):
    data = load_breast_cancer()
    dtrain = xgb.DMatrix(data.data, label=data.target)
    param = {}
    param['objective'] = 'binary:logistic'
    param.update(param_in)
    res_tmp = {}
    res = {}
    num_rounds = 10
    xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[param['tree_method']] = res_tmp['train']['error']
    param["tree_method"] = comparison_tree_method
    xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[comparison_tree_method] = res_tmp['train']['error']
    return res


def train_sparse(param_in, comparison_tree_method):
    n = 5000
    sparsity = 0.75
    X, y = make_regression(n, random_state=rng)
    X = np.array([[np.nan if rng.uniform(0, 1) < sparsity else x for x in x_row] for x_row in X])
    dtrain = xgb.DMatrix(X, label=y)
    param = {}
    param.update(param_in)
    res_tmp = {}
    res = {}
    num_rounds = 10
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[param['tree_method']] = res_tmp['train']['rmse']
    param["tree_method"] = comparison_tree_method
    bst = xgb.train(param, dtrain, num_rounds, [(dtrain, 'train')], evals_result=res_tmp)
    res[comparison_tree_method] = res_tmp['train']['rmse']
    return res


# Enumerates all permutations of variable parameters
def assert_updater_accuracy(tree_method, comparison_tree_method, variable_param, tolerance):
    param = {'tree_method': tree_method }
    names = sorted(variable_param)
    combinations = it.product(*(variable_param[Name] for Name in names))

    for set in combinations:
        print(names, file=sys.stderr)
        print(set, file=sys.stderr)
        param_tmp = param.copy()
        for i, name in enumerate(names):
            param_tmp[name] = set[i]

        print(param_tmp, file=sys.stderr)
        assert_accuracy(train_boston(param_tmp, comparison_tree_method), tree_method, comparison_tree_method, tolerance, param_tmp)
        assert_accuracy(train_digits(param_tmp, comparison_tree_method), tree_method, comparison_tree_method, tolerance, param_tmp)
        assert_accuracy(train_cancer(param_tmp, comparison_tree_method), tree_method, comparison_tree_method, tolerance, param_tmp)
        assert_accuracy(train_sparse(param_tmp, comparison_tree_method), tree_method, comparison_tree_method, tolerance, param_tmp)


@attr('gpu')
class TestGPU(unittest.TestCase):
    def test_gpu_hist(self):
        variable_param = {'max_depth': [2, 6, 11], 'max_bin': [2, 16, 1024], 'n_gpus': [1, -1]}
        assert_updater_accuracy('gpu_hist', 'hist', variable_param, 0.02)

    def test_gpu_exact(self):
        variable_param = {'max_depth': [2, 6, 15]}
        assert_updater_accuracy('gpu_exact', 'exact', variable_param, 0.02)

    def test_gpu_hist_experimental(self):
        variable_param = {'n_gpus': [1, -1], 'max_depth': [2, 6], 'max_leaves': [255, 4], 'max_bin': [2, 16, 1024]}
        assert_updater_accuracy('gpu_hist_experimental', 'hist', variable_param, 0.01)
