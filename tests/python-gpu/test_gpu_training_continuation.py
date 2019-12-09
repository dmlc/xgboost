import unittest
import numpy as np
import xgboost as xgb
import json

rng = np.random.RandomState(1994)


class TestGPUTrainingContinuation(unittest.TestCase):
    def test_training_continuation_binary(self):
        kRows = 32
        kCols = 16
        X = np.random.randn(kRows, kCols)
        y = np.random.randn(kRows)
        dtrain = xgb.DMatrix(X, y)
        params = {'tree_method': 'gpu_hist', 'max_depth': '2'}
        bst_0 = xgb.train(params, dtrain, num_boost_round=4)
        dump_0 = bst_0.get_dump(dump_format='json')

        bst_1 = xgb.train(params, dtrain, num_boost_round=2)
        bst_1 = xgb.train(params, dtrain, num_boost_round=2, xgb_model=bst_1)
        dump_1 = bst_1.get_dump(dump_format='json')

        def recursive_compare(obj_0, obj_1):
            if isinstance(obj_0, float):
                assert np.isclose(obj_0, obj_1)
            elif isinstance(obj_0, str):
                assert obj_0 == obj_1
            elif isinstance(obj_0, int):
                assert obj_0 == obj_1
            elif isinstance(obj_0, dict):
                keys_0 = list(obj_0.keys())
                keys_1 = list(obj_1.keys())
                values_0 = list(obj_0.values())
                values_1 = list(obj_1.values())
                for i in range(len(obj_0.items())):
                    assert keys_0[i] == keys_1[i]
                    if list(obj_0.keys())[i] != 'missing':
                        recursive_compare(values_0[i],
                                          values_1[i])
            else:
                for i in range(len(obj_0)):
                    recursive_compare(obj_0[i], obj_1[i])

        for i in range(len(dump_0)):
            obj_0 = json.loads(dump_0[i])
            obj_1 = json.loads(dump_1[i])
            recursive_compare(obj_0, obj_1)
