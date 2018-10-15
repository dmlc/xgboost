import numpy as np
import sys
import unittest

sys.path.append("tests/python")
import xgboost as xgb
from regression_test_utilities import run_suite, parameter_combinations, \
    assert_results_non_increasing
from nose.plugins.attrib import attr

def assert_gpu_results(cpu_results, gpu_results):
    for cpu_res, gpu_res in zip(cpu_results, gpu_results):
        # Check final eval result roughly equivalent
        assert np.allclose(cpu_res["eval"][-1], gpu_res["eval"][-1], 1e-2, 1e-2)

datasets = ["Boston", "Cancer", "Digits", "Sparse regression",
            "Sparse regression with weights", "Small weights regression"]

class TestGPU(unittest.TestCase):
    def test_gpu_exact(self):
        variable_param = {'max_depth': [2, 6, 15], }
        for param in parameter_combinations(variable_param):
            param['tree_method'] = 'gpu_exact'
            gpu_results = run_suite(param, select_datasets=datasets)
            assert_results_non_increasing(gpu_results, 1e-2)
            param['tree_method'] = 'exact'
            cpu_results = run_suite(param, select_datasets=datasets)
            assert_gpu_results(cpu_results, gpu_results)

    def test_gpu_hist(self):
        variable_param = {'n_gpus': [-1], 'max_depth': [2, 8], 'max_leaves': [255, 4],
                          'max_bin': [2, 256], 'min_child_weight': [0, 1], 'lambda': [0.0, 1.0],
                          'grow_policy': ['lossguide']}
        for param in parameter_combinations(variable_param):
            param['tree_method'] = 'gpu_hist'
            gpu_results = run_suite(param, select_datasets=datasets)
            assert_results_non_increasing(gpu_results, 1e-2)
            param['tree_method'] = 'hist'
            cpu_results = run_suite(param, select_datasets=datasets)
            assert_gpu_results(cpu_results, gpu_results)

    @attr('mgpu')
    def test_gpu_hist_mgpu(self):
        variable_param = {'n_gpus': [-1], 'max_depth': [2, 10], 'max_leaves': [255, 4],
                          'max_bin': [2, 256],
                          'grow_policy': ['lossguide']}
        for param in parameter_combinations(variable_param):
            param['tree_method'] = 'gpu_hist'
            gpu_results = run_suite(param, select_datasets=datasets)
            assert_results_non_increasing(gpu_results, 1e-2)
            # FIXME: re-enable next three lines, to compare against CPU
            #param['tree_method'] = 'hist'
            #cpu_results = run_suite(param, select_datasets=datasets)
            #assert_gpu_results(cpu_results, gpu_results)
