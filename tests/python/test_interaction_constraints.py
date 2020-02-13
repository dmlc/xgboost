# -*- coding: utf-8 -*-
import numpy as np
import xgboost
import unittest
import testing as tm
import pytest

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestInteractionConstraints(unittest.TestCase):
    def run_interaction_constraints(self, tree_method):
        x1 = np.random.normal(loc=1.0, scale=1.0, size=1000)
        x2 = np.random.normal(loc=1.0, scale=1.0, size=1000)
        x3 = np.random.choice([1, 2, 3], size=1000, replace=True)
        y = x1 + x2 + x3 + x1 * x2 * x3 \
            + np.random.normal(
                loc=0.001, scale=1.0, size=1000) + 3 * np.sin(x1)
        X = np.column_stack((x1, x2, x3))
        dtrain = xgboost.DMatrix(X, label=y)

        params = {
            'max_depth': 3,
            'eta': 0.1,
            'nthread': 2,
            'interaction_constraints': '[[0, 1]]',
            'tree_method': tree_method
        }
        num_boost_round = 12
        # Fit a model that only allows interaction between x1 and x2
        bst = xgboost.train(
            params, dtrain, num_boost_round, evals=[(dtrain, 'train')])

        # Set all observations to have the same x3 values then increment
        #   by the same amount
        def f(x):
            tmat = xgboost.DMatrix(
                np.column_stack((x1, x2, np.repeat(x, 1000))))
            return bst.predict(tmat)

        preds = [f(x) for x in [1, 2, 3]]

        # Check incrementing x3 has the same effect on all observations
        #   since x3 is constrained to be independent of x1 and x2
        #   and all observations start off from the same x3 value
        diff1 = preds[1] - preds[0]
        assert np.all(np.abs(diff1 - diff1[0]) < 1e-4)
        diff2 = preds[2] - preds[1]
        assert np.all(np.abs(diff2 - diff2[0]) < 1e-4)

    def test_exact_interaction_constraints(self):
        self.run_interaction_constraints(tree_method='exact')

    def test_hist_interaction_constraints(self):
        self.run_interaction_constraints(tree_method='hist')

    def test_approx_interaction_constraints(self):
        self.run_interaction_constraints(tree_method='approx')

    @pytest.mark.skipif(**tm.no_sklearn())
    def training_accuracy(self, tree_method):
        from sklearn.metrics import accuracy_score
        dtrain = xgboost.DMatrix(dpath + 'agaricus.txt.train?indexing_mode=1')
        dtest = xgboost.DMatrix(dpath + 'agaricus.txt.test?indexing_mode=1')
        params = {
            'eta': 1,
            'max_depth': 6,
            'objective': 'binary:logistic',
            'tree_method': tree_method,
            'interaction_constraints': '[[1,2], [2,3,4]]'
        }
        num_boost_round = 5

        params['grow_policy'] = 'lossguide'
        bst = xgboost.train(params, dtrain, num_boost_round)
        pred_dtest = (bst.predict(dtest) < 0.5)
        assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1

        params['grow_policy'] = 'depthwise'
        bst = xgboost.train(params, dtrain, num_boost_round)
        pred_dtest = (bst.predict(dtest) < 0.5)
        assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1

    def test_hist_training_accuracy(self):
        self.training_accuracy(tree_method='hist')

    def test_exact_training_accuracy(self):
        self.training_accuracy(tree_method='exact')

    def test_approx_training_accuracy(self):
        self.training_accuracy(tree_method='approx')
