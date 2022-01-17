import numpy as np
import gc
import pytest
import xgboost as xgb
from hypothesis import given, strategies, assume, settings, note

import sys
sys.path.append("tests/python")
import testing as tm

parameter_strategy = strategies.fixed_dictionaries({
    'max_depth': strategies.integers(0, 11),
    'max_leaves': strategies.integers(0, 256),
    'max_bin': strategies.integers(2, 1024),
    'grow_policy': strategies.sampled_from(['lossguide', 'depthwise']),
    'single_precision_histogram': strategies.booleans(),
    'min_child_weight': strategies.floats(0.5, 2.0),
    'seed': strategies.integers(0, 10),
    # We cannot enable subsampling as the training loss can increase
    # 'subsample': strategies.floats(0.5, 1.0),
    'colsample_bytree': strategies.floats(0.5, 1.0),
    'colsample_bylevel': strategies.floats(0.5, 1.0),
}).filter(lambda x: (x['max_depth'] > 0 or x['max_leaves'] > 0) and (
    x['max_depth'] > 0 or x['grow_policy'] == 'lossguide'))


def train_result(param, dmat, num_rounds):
    result = {}
    xgb.train(param, dmat, num_rounds, [(dmat, 'train')], verbose_eval=False,
              evals_result=result)
    return result


class TestOneAPIUpdaters:
    @given(parameter_strategy, strategies.integers(1, 5),
           tm.dataset_strategy.filter(lambda x: x.name != "empty"))
    @settings(deadline=None)
    def test_oneapi_hist(self, param, num_rounds, dataset):
        param['updater'] = 'grow_quantile_histmaker_oneapi'
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result['train'][dataset.metric])

    @given(tm.dataset_strategy.filter(lambda x: x.name != "empty"), strategies.integers(0, 1))
    @settings(deadline=None)
    def test_specified_device_id_oneapi_update(self, dataset, device_id):
        param = {'updater': 'grow_quantile_histmaker_oneapi', 'device_id': device_id}
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), 10)
        assert tm.non_increasing(result['train'][dataset.metric])
