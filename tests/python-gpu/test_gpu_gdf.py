import unittest
import pytest
import testing as tm
import numpy as np
from sklearn import datasets
import sys
import xgboost as xgb
from regression_test_utilities import run_suite, parameter_combinations, \
    assert_results_non_increasing, Dataset

try:
    import cudf.dataframe as cudf
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    tm.no_cudf()['condition'],
    reason=tm.no_cudf()['reason'])


def get_cudf():
    rng = np.random.RandomState(199)
    n = 50000
    m = 20
    X, y = datasets.make_regression(n, m, random_state=rng)
    Xy = (np.ascontiguousarray
          (np.transpose(np.concatenate((X, np.expand_dims(y, axis=1)), axis=1))))
    df = cudf.DataFrame(list(zip(['col%d' % i for i in range(m + 1)], Xy)))
    all_columns = list(df.columns)
    cols_X = all_columns[0:len(all_columns) - 1]
    cols_y = [all_columns[len(all_columns) - 1]]
    return df[cols_X], df[cols_y]


class TestCudf(unittest.TestCase):
    cudf_datasets = [Dataset("GDF", get_cudf, "reg:linear", "rmse")]

    def test_cudf(self):
        variable_param = {'n_gpus': [1], 'max_depth': [10], 'max_leaves': [255],
                          'max_bin': [255],
                          'grow_policy': ['lossguide']}
        for param in parameter_combinations(variable_param):
            param['tree_method'] = 'gpu_hist'
            gpu_results = run_suite(param, num_rounds=20,
                                    select_datasets=self.cudf_datasets)
            assert_results_non_increasing(gpu_results, 1e-2)

    def test_set_info_single_column(self):
        X, y = get_cudf()
        y = y[:, 0]
        dtrain = xgb.DMatrix(X, y)
        dtrain.set_float_info("weight", y)
        dtrain.set_base_margin(y)
