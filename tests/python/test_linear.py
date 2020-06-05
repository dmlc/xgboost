import testing as tm
from hypothesis import strategies, given, settings, note
import xgboost as xgb

parameter_strategy = strategies.fixed_dictionaries({
    'booster': strategies.just('gblinear'),
    'eta': strategies.floats(0.01, 0.5),
    'tolerance': strategies.floats(1e-5, 1e-2),
    'nthread': strategies.integers(0, 4),
    'alpha': strategies.floats(1e-5, 1.0),
    'lambda': strategies.floats(1e-5, 1.0)
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
    @given(parameter_strategy, strategies.integers(1, 50),
           tm.dataset_strategy, coord_strategy)
    @settings(deadline=2000)
    def test_coordinate(self, param, num_rounds, dataset, coord_param):
        param['updater'] = 'coord_descent'
        param.update(coord_param)
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        assert tm.non_increasing(result['train'][dataset.metric])

    @given(parameter_strategy, strategies.integers(1, 50),
           tm.dataset_strategy)
    @settings(deadline=2000)
    def test_shotgun(self, param, num_rounds, dataset):
        param['updater'] = 'shotgun'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result['train'][dataset.metric], 1e-2)
