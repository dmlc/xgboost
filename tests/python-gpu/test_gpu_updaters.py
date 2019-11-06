import numpy as np
import sys
import unittest
import pytest
import xgboost

sys.path.append("tests/python")
from regression_test_utilities import run_suite, parameter_combinations, \
    assert_results_non_increasing


def assert_gpu_results(cpu_results, gpu_results):
    for cpu_res, gpu_res in zip(cpu_results, gpu_results):
        # Check final eval result roughly equivalent
        assert np.allclose(cpu_res["eval"][-1],
                           gpu_res["eval"][-1], 1e-2, 1e-2)


datasets = ["Boston", "Cancer", "Digits", "Sparse regression",
            "Sparse regression with weights", "Small weights regression"]


class TestGPU(unittest.TestCase):
    def test_gpu_hist(self):
        test_param = parameter_combinations({'gpu_id': [0],
                                             'max_depth': [2, 8],
                                             'max_leaves': [255, 4],
                                             'max_bin': [2, 256],
                                             'grow_policy': ['lossguide']})
        test_param.append({'single_precision_histogram': True})
        test_param.append({'min_child_weight': 0,
                           'lambda': 0})
        for param in test_param:
            param['tree_method'] = 'gpu_hist'
            gpu_results = run_suite(param, select_datasets=datasets)
            assert_results_non_increasing(gpu_results, 1e-2)
            param['tree_method'] = 'hist'
            cpu_results = run_suite(param, select_datasets=datasets)
            assert_gpu_results(cpu_results, gpu_results)

    def test_with_empty_dmatrix(self):
        # FIXME(trivialfis): This should be done with all updaters
        kRows = 0
        kCols = 100

        X = np.empty((kRows, kCols))
        y = np.empty((kRows))

        dtrain = xgboost.DMatrix(X, y)

        bst = xgboost.train({'verbosity': 2,
                             'tree_method': 'gpu_hist',
                             'gpu_id': 0},
                            dtrain,
                            verbose_eval=True,
                            num_boost_round=6,
                            evals=[(dtrain, 'Train')])

        kRows = 100
        X = np.random.randn(kRows, kCols)

        dtest = xgboost.DMatrix(X)
        predictions = bst.predict(dtest)
        np.testing.assert_allclose(predictions, 0.5, 1e-6)

    @pytest.mark.mgpu
    def test_specified_gpu_id_gpu_update(self):
        variable_param = {'gpu_id': [1],
                          'max_depth': [8],
                          'max_leaves': [255, 4],
                          'max_bin': [2, 64],
                          'grow_policy': ['lossguide'],
                          'tree_method': ['gpu_hist']}
        for param in parameter_combinations(variable_param):
            gpu_results = run_suite(param, select_datasets=datasets)
            assert_results_non_increasing(gpu_results, 1e-2)
