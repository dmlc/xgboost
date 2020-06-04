import sys
import pytest
import unittest

sys.path.append('tests/python/')
import test_linear  # noqa: E402
import testing as tm  # noqa: E402


class TestGPULinear(unittest.TestCase):
    datasets = ["Boston", "Digits", "Cancer", "Sparse regression"]
    common_param = {
        'booster': ['gblinear'],
        'updater': ['gpu_coord_descent'],
        'eta': [0.5],
        'top_k': [10],
        'tolerance': [1e-5],
        'alpha': [.1],
        'lambda': [0.005],
        'coordinate_selection': ['cyclic', 'random', 'greedy']}

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_gpu_coordinate(self):
        parameters = self.common_param.copy()
        parameters['gpu_id'] = [0]
        for param in test_linear.parameter_combinations(parameters):
            results = test_linear.run_suite(
                param, 100, self.datasets, scale_features=True)
            test_linear.assert_regression_result(results, 1e-2)
            test_linear.assert_classification_result(results)
