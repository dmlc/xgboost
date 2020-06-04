import numpy as np
import sys
import unittest
import pytest
import xgboost as xgb

sys.path.append("tests/python")
import testing as tm
from regression_test_utilities import run_suite, parameter_combinations, \
    assert_results_non_increasing


def assert_gpu_results(cpu_results, gpu_results):
    for cpu_res, gpu_res in zip(cpu_results, gpu_results):
        # Check final eval result roughly equivalent
        assert np.allclose(cpu_res["eval"][-1],
                           gpu_res["eval"][-1], 1e-1, 1e-1)


datasets = ["Boston", "Cancer", "Digits", "Sparse regression",
            "Sparse regression with weights", "Small weights regression"]

test_param = parameter_combinations({
    'gpu_id': [0],
    'max_depth': [2, 8],
    'max_leaves': [255, 4],
    'max_bin': [4, 256],
    'grow_policy': ['lossguide'],
    'single_precision_histogram': [True],
    'min_child_weight': [0],
    'lambda': [0]})


class TestGPU(unittest.TestCase):
    def test_gpu_hist(self):
        for param in test_param:
            param['tree_method'] = 'gpu_hist'
            gpu_results = run_suite(param, select_datasets=datasets)
            assert_results_non_increasing(gpu_results, 1e-2)
            param['tree_method'] = 'hist'
            cpu_results = run_suite(param, select_datasets=datasets)
            assert_gpu_results(cpu_results, gpu_results)

    @pytest.mark.skipif(**tm.no_cupy())
    def test_gpu_hist_device_dmatrix(self):
        # DeviceDMatrix does not currently accept sparse formats
        device_dmatrix_datasets = ["Boston", "Cancer", "Digits"]
        for param in test_param:
            param['tree_method'] = 'gpu_hist'
            
            gpu_results_device_dmatrix = run_suite(param, select_datasets=device_dmatrix_datasets,
                                                   DMatrixT=xgb.DeviceQuantileDMatrix,
                                                   dmatrix_params={'max_bin': param['max_bin']})
            assert_results_non_increasing(gpu_results_device_dmatrix, 1e-2)
            gpu_results = run_suite(param, select_datasets=device_dmatrix_datasets)
            assert_gpu_results(gpu_results, gpu_results_device_dmatrix)

    # NOTE(rongou): Because the `Boston` dataset is too small, this only tests external memory mode
    # with a single page. To test multiple pages, set DMatrix::kPageSize to, say, 1024.
    def test_external_memory(self):
        for param in reversed(test_param):
            param['tree_method'] = 'gpu_hist'
            param['gpu_page_size'] = 1024
            gpu_results = run_suite(param, select_datasets=["Boston"])
            assert_results_non_increasing(gpu_results, 1e-2)
            ext_mem_results = run_suite(param, select_datasets=["Boston External Memory"])
            assert_results_non_increasing(ext_mem_results, 1e-2)
            assert_gpu_results(gpu_results, ext_mem_results)
            break

    def test_with_empty_dmatrix(self):
        # FIXME(trivialfis): This should be done with all updaters
        kRows = 0
        kCols = 100

        X = np.empty((kRows, kCols))
        y = np.empty((kRows))

        dtrain = xgb.DMatrix(X, y)

        bst = xgb.train({'verbosity': 2,
                         'tree_method': 'gpu_hist',
                         'gpu_id': 0},
                        dtrain,
                        verbose_eval=True,
                        num_boost_round=6,
                        evals=[(dtrain, 'Train')])

        kRows = 100
        X = np.random.randn(kRows, kCols)

        dtest = xgb.DMatrix(X)
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
