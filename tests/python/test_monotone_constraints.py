import numpy as np
import xgboost as xgb
import testing as tm
import pytest

dpath = 'demo/data/'


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


class TestMonotoneConstraints:

    @pytest.mark.parametrize('tree_method,extra_kwargs',
                             [('exact', {}), ('hist', {}), ('hist', {'grow_policy': 'lossguide'})],
                             ids=['exact', 'hist-depthwise', 'hist-lossguide'])
    def test_monotone_constraints(self, tree_method, extra_kwargs):
        params = {
            'tree_method': tree_method, 'verbosity': 1, 'monotone_constraints': '(1, -1)'
        }
        params.update(extra_kwargs)
        bst = xgb.train(params, training_dset)
        assert is_correctly_constrained(bst)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize('grow_policy', ['lossguide', 'depthwise'])
    def test_training_accuracy(self, grow_policy):
        from sklearn.metrics import accuracy_score
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train?indexing_mode=1')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test?indexing_mode=1')
        params = {'eta': 1, 'max_depth': 6, 'objective': 'binary:logistic',
                  'tree_method': 'hist', 'grow_policy': grow_policy,
                  'monotone_constraints': '(1, 0)'}
        num_boost_round = 5
        bst = xgb.train(params, dtrain, num_boost_round)
        pred_dtest = (bst.predict(dtest) < 0.5)
        assert accuracy_score(dtest.get_label(), pred_dtest) < 0.1
