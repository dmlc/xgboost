import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

dpath = 'demo/data/'


def is_increasing(y):
    return np.count_nonzero(np.diff(y) < 0.0) == 0


def is_decreasing(y):
    return np.count_nonzero(np.diff(y) > 0.0) == 0


def is_correctly_constrained(learner, feature_names=None):
    n = 100
    variable_x = np.linspace(0, 1, n).reshape((n, 1))
    fixed_xs_values = np.linspace(0, 1, n)

    for i in range(n):
        fixed_x = fixed_xs_values[i] * np.ones((n, 1))
        monotonically_increasing_x = np.column_stack((variable_x, fixed_x))
        monotonically_increasing_dset = xgb.DMatrix(monotonically_increasing_x,
                                                    feature_names=feature_names)
        monotonically_increasing_y = learner.predict(
            monotonically_increasing_dset
        )

        monotonically_decreasing_x = np.column_stack((fixed_x, variable_x))
        monotonically_decreasing_dset = xgb.DMatrix(monotonically_decreasing_x,
                                                    feature_names=feature_names)
        monotonically_decreasing_y = learner.predict(
            monotonically_decreasing_dset
        )

        if not (
            is_increasing(monotonically_increasing_y) and
            is_decreasing(monotonically_decreasing_y)
        ):
            return False

    return True


number_of_dpoints = 1000
x1_positively_correlated_with_y = np.random.random(size=number_of_dpoints)
x2_negatively_correlated_with_y = np.random.random(size=number_of_dpoints)

x = np.column_stack((
    x1_positively_correlated_with_y, x2_negatively_correlated_with_y
))
zs = np.random.normal(loc=0.0, scale=0.01, size=number_of_dpoints)
y = (
    5 * x1_positively_correlated_with_y +
    np.sin(10 * np.pi * x1_positively_correlated_with_y) -
    5 * x2_negatively_correlated_with_y -
    np.cos(10 * np.pi * x2_negatively_correlated_with_y) +
    zs
)
training_dset = xgb.DMatrix(x, label=y)


class TestMonotoneConstraints:
    def test_monotone_constraints_for_exact_tree_method(self):

        # first check monotonicity for the 'exact' tree method
        params_for_constrained_exact_method = {
            'tree_method': 'exact', 'verbosity': 1,
            'monotone_constraints': '(1, -1)'
        }
        constrained_exact_method = xgb.train(
            params_for_constrained_exact_method, training_dset
        )
        assert is_correctly_constrained(constrained_exact_method)

    @pytest.mark.parametrize(
        "tree_method,policy",
        [
            ("hist", "depthwise"),
            ("approx", "depthwise"),
            ("hist", "lossguide"),
            ("approx", "lossguide"),
        ],
    )
    def test_monotone_constraints(self, tree_method: str, policy: str) -> None:
        params_for_constrained = {
            "tree_method": tree_method,
            "grow_policy": policy,
            "monotone_constraints": "(1, -1)",
        }
        constrained = xgb.train(params_for_constrained, training_dset)
        assert is_correctly_constrained(constrained)

    def test_monotone_constraints_tuple(self) -> None:
        params_for_constrained = {"monotone_constraints": (1, -1)}
        constrained = xgb.train(params_for_constrained, training_dset)
        assert is_correctly_constrained(constrained)

    @pytest.mark.parametrize('format', [dict, list])
    def test_monotone_constraints_feature_names(self, format):

        # next check monotonicity when initializing monotone_constraints by feature names
        params = {
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'monotone_constraints': {'feature_0': 1, 'feature_1': -1}
        }

        if format == list:
            params = list(params.items())

        with pytest.raises(ValueError):
            xgb.train(params, training_dset)

        feature_names = ['feature_0', 'feature_2']
        training_dset_w_feature_names = xgb.DMatrix(x, label=y, feature_names=feature_names)

        with pytest.raises(ValueError):
            xgb.train(params, training_dset_w_feature_names)

        feature_names = ['feature_0', 'feature_1']
        training_dset_w_feature_names = xgb.DMatrix(x, label=y, feature_names=feature_names)

        constrained_learner = xgb.train(
            params, training_dset_w_feature_names
        )

        assert is_correctly_constrained(constrained_learner, feature_names)


    @pytest.mark.skipif(**tm.no_sklearn())
    def test_training_accuracy(self):
        from sklearn.metrics import accuracy_score
        dtrain = xgb.DMatrix(dpath + "agaricus.txt.train?indexing_mode=1&format=libsvm")
        dtest = xgb.DMatrix(dpath + "agaricus.txt.test?indexing_mode=1&format=libsvm")
        params = {'eta': 1, 'max_depth': 6, 'objective': 'binary:logistic',
                  'tree_method': 'hist', 'monotone_constraints': '(1, 0)'}
        num_boost_round = 5

        params['grow_policy'] = 'lossguide'
        bst = xgb.train(params, dtrain, num_boost_round)
        pred_dtest = (bst.predict(dtest) < 0.5)
        assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1

        params['grow_policy'] = 'depthwise'
        bst = xgb.train(params, dtrain, num_boost_round)
        pred_dtest = (bst.predict(dtest) < 0.5)
        assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1
