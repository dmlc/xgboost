import numpy as np
import xgboost as xgb
import unittest


def is_increasing(y):
    return np.count_nonzero(np.diff(y) < 0.0) == 0


def is_decreasing(y):
    return np.count_nonzero(np.diff(y) > 0.0) == 0


def is_correctly_constrained(learner):
    n = 100
    variable_x = np.linspace(0, 1, n).reshape((n, 1))
    fixed_xs_values = np.linspace(0, 1, n)

    for i in range(n):
        fixed_x = fixed_xs_values[i] * np.ones((n, 1))
        monotonically_increasing_x = np.column_stack((variable_x, fixed_x))
        monotonically_increasing_dset = xgb.DMatrix(monotonically_increasing_x)
        monotonically_increasing_y = learner.predict(
            monotonically_increasing_dset
        )

        monotonically_decreasing_x = np.column_stack((fixed_x, variable_x))
        monotonically_decreasing_dset = xgb.DMatrix(monotonically_decreasing_x)
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


class TestMonotoneConstraints(unittest.TestCase):

    def test_monotone_constraints_for_exact_tree_method(self):

        # first check monotonicity for the 'exact' tree method
        params_for_constrained_exact_method = {
            'tree_method': 'exact', 'silent': 1,
            'monotone_constraints': '(1, -1)'
        }
        constrained_exact_method = xgb.train(
            params_for_constrained_exact_method, training_dset
        )
        assert is_correctly_constrained(constrained_exact_method)

    def test_monotone_constraints_for_hist_tree_method(self):

        # next check monotonicity for the 'hist' tree method
        params_for_constrained_hist_method = {
            'tree_method': 'hist', 'silent': 1,
            'monotone_constraints': '(1, -1)'
        }
        constrained_hist_method = xgb.train(
            params_for_constrained_hist_method, training_dset
        )

        assert is_correctly_constrained(constrained_hist_method)
