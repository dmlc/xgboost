import pytest
from hypothesis import assume, given, note, settings, strategies

import xgboost as xgb
from xgboost import testing as tm

pytestmark = tm.timeout(10)

parameter_strategy = strategies.fixed_dictionaries({
    'booster': strategies.just('gblinear'),
    'eta': strategies.floats(0.01, 0.25),
    'tolerance': strategies.floats(1e-5, 1e-2),
    'nthread': strategies.integers(1, 4),
    'feature_selector': strategies.sampled_from(['cyclic', 'shuffle',
                                                 'greedy', 'thrifty']),
    'top_k': strategies.integers(1, 10),
})


def train_result(param, dmat, num_rounds):
    result = {}
    booster = xgb.train(
        param, dmat, num_rounds, [(dmat, 'train')], verbose_eval=False,
        evals_result=result
    )
    assert booster.num_boosted_rounds() == num_rounds
    return result


class TestGPULinear:
    @given(parameter_strategy, strategies.integers(10, 50), tm.make_dataset_strategy())
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_gpu_coordinate(self, param, num_rounds, dataset):
        assume(len(dataset.y) > 0)
        param['updater'] = 'gpu_coord_descent'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)['train'][dataset.metric]
        note(result)
        assert tm.non_increasing(result)

    # Loss is not guaranteed to always decrease because of regularisation parameters
    # We test a weaker condition that the loss has not increased between the first and last
    # iteration
    @given(
        parameter_strategy,
        strategies.integers(10, 50),
        tm.make_dataset_strategy(),
        strategies.floats(1e-5, 0.8),
        strategies.floats(1e-5, 0.8)
    )
    @settings(deadline=None, max_examples=20, print_blob=True)
    def test_gpu_coordinate_regularised(self, param, num_rounds, dataset, alpha, lambd):
        assume(len(dataset.y) > 0)
        param['updater'] = 'gpu_coord_descent'
        param['alpha'] = alpha
        param['lambda'] = lambd
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)['train'][dataset.metric]
        note(result)
        assert tm.non_increasing([result[0], result[-1]])

    @pytest.mark.skipif(**tm.no_cupy())
    def test_gpu_coordinate_from_cupy(self):
        # Training linear model is quite expensive, so we don't include it in
        # test_from_cupy.py
        import cupy
        params = {'booster': 'gblinear', 'updater': 'gpu_coord_descent',
                  'n_estimators': 100}
        X, y = tm.get_california_housing()
        cpu_model = xgb.XGBRegressor(**params)
        cpu_model.fit(X, y)
        cpu_predt = cpu_model.predict(X)

        X = cupy.array(X)
        y = cupy.array(y)
        gpu_model = xgb.XGBRegressor(**params)
        gpu_model.fit(X, y)
        gpu_predt = gpu_model.predict(X)
        cupy.testing.assert_allclose(cpu_predt, gpu_predt)
