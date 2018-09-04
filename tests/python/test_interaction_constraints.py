# -*- coding: utf-8 -*-
import numpy as np
import xgboost
import unittest

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestInteractionConstraints(unittest.TestCase):

    def test_interaction_constraints(self):
        x1 = np.random.normal(loc=1.0, scale=1.0, size=1000)
        x2 = np.random.normal(loc=1.0, scale=1.0, size=1000)
        x3 = np.random.choice([1, 2, 3], size=1000, replace=True)
        y = x1 + x2 + x3 + x1 * x2 * x3 \
            + np.random.normal(loc=0.001, scale=1.0, size=1000) + 3 * np.sin(x1)
        X = np.column_stack((x1, x2, x3))
        dtrain = xgboost.DMatrix(X, label=y)

        params = {'max_depth': 3, 'eta': 0.1, 'nthread': 2, 'silent': 1,
                  'interaction_constraints': '[[0, 1]]'}
        num_boost_round = 100
        # Fit a model that only allows interaction between x1 and x2
        bst = xgboost.train(params, dtrain, num_boost_round, evals=[(dtrain, 'train')])

        # Set all observations to have the same x3 values then increment
        #   by the same amount
        def f(x):
            tmat = xgboost.DMatrix(np.column_stack((x1, x2, np.repeat(x, 1000))))
            return bst.predict(tmat)
        preds = [f(x) for x in [1, 2, 3]]

        # Check incrementing x3 has the same effect on all observations
        #   since x3 is constrained to be independent of x1 and x2
        #   and all observations start off from the same x3 value
        diff1 = preds[1] - preds[0]
        assert np.all(np.abs(diff1 - diff1[0]) < 1e-4)
        diff2 = preds[2] - preds[1]
        assert np.all(np.abs(diff2 - diff2[0]) < 1e-4)
