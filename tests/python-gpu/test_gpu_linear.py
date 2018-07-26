import sys

sys.path.append('tests/python/')
import test_linear
import testing as tm
import unittest


class TestGPULinear(unittest.TestCase):

    datasets = ["Boston", "Digits", "Cancer", "Sparse regression",
                "Boston External Memory"]
    
    def test_gpu_coordinate(self):
        tm._skip_if_no_sklearn()
        variable_param = {
            'booster': ['gblinear'],
            'updater': ['coord_descent'],
            'eta': [0.5],
            'top_k': [10],
            'tolerance': [1e-5],
            'nthread': [2],
            'alpha': [.005, .1],
            'lambda': [0.005],
            'coordinate_selection': ['cyclic', 'random', 'greedy'],
            'n_gpus': [-1]
        }
        for param in test_linear.parameter_combinations(variable_param):
            results = test_linear.run_suite(
                param, 200, self.datasets, scale_features=True)
            test_linear.assert_regression_result(results, 1e-2)
            test_linear.assert_classification_result(results)
