from hypothesis import given, note, settings, strategies

import xgboost as xgb
from xgboost import testing as tm

pytestmark = tm.timeout(20)


parameter_strategy = strategies.fixed_dictionaries({
    'booster': strategies.just('gblinear'),
    'eta': strategies.floats(0.01, 0.25),
    'tolerance': strategies.floats(1e-5, 1e-2),
    'nthread': strategies.integers(1, 4),
})

coord_strategy = strategies.fixed_dictionaries({
    'feature_selector': strategies.sampled_from(['cyclic', 'shuffle',
                                                 'greedy', 'thrifty']),
    'top_k': strategies.integers(1, 10),
})


def train_result(param, dmat, num_rounds):
    result = {}
    xgb.train(param, dmat, num_rounds, [(dmat, 'train')], verbose_eval=False,
              evals_result=result)
    return result


class TestLinear:
    @given(
        parameter_strategy,
        strategies.integers(10, 50),
        tm.make_dataset_strategy(),
        coord_strategy
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_coordinate(self, param, num_rounds, dataset, coord_param):
        param['updater'] = 'coord_descent'
        param.update(coord_param)
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)['train'][dataset.metric]
        note(result)
        assert tm.non_increasing(result, 5e-4)

    # Loss is not guaranteed to always decrease because of regularisation parameters
    # We test a weaker condition that the loss has not increased between the first and last
    # iteration
    @given(
        parameter_strategy,
        strategies.integers(10, 50),
        tm.make_dataset_strategy(),
        coord_strategy,
        strategies.floats(1e-5, 0.8),
        strategies.floats(1e-5, 0.8)
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_coordinate_regularised(self, param, num_rounds, dataset, coord_param, alpha, lambd):
        param['updater'] = 'coord_descent'
        param['alpha'] = alpha
        param['lambda'] = lambd
        param.update(coord_param)
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)['train'][dataset.metric]
        note(result)
        assert tm.non_increasing([result[0], result[-1]])

    @given(
        parameter_strategy, strategies.integers(10, 50), tm.make_dataset_strategy()
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_shotgun(self, param, num_rounds, dataset):
        param['updater'] = 'shotgun'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)['train'][dataset.metric]
        note(result)
        # shotgun is non-deterministic, so we relax the test by only using first and last
        # iteration.
        if len(result) > 2:
            sampled_result = (result[0], result[-1])
        else:
            sampled_result = result
        assert tm.non_increasing(sampled_result)

    @given(
        parameter_strategy,
        strategies.integers(10, 50),
        tm.make_dataset_strategy(),
        strategies.floats(1e-5, 1.0),
        strategies.floats(1e-5, 1.0)
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_shotgun_regularised(self, param, num_rounds, dataset, alpha, lambd):
        param['updater'] = 'shotgun'
        param['alpha'] = alpha
        param['lambda'] = lambd
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)['train'][dataset.metric]
        note(result)
        assert tm.non_increasing([result[0], result[-1]])
