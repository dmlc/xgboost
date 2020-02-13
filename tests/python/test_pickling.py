import pickle
import numpy as np
import xgboost as xgb
import os
import unittest


kRows = 100
kCols = 10


def generate_data():
    X = np.random.randn(kRows, kCols)
    y = np.random.randn(kRows)
    return X, y


class TestPickling(unittest.TestCase):
    def run_model_pickling(self, xgb_params):
        X, y = generate_data()
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(xgb_params, dtrain)

        dump_0 = bst.get_dump(dump_format='json')
        assert dump_0

        filename = 'model.pkl'

        with open(filename, 'wb') as fd:
            pickle.dump(bst, fd)

        with open(filename, 'rb') as fd:
            bst = pickle.load(fd)

        with open(filename, 'wb') as fd:
            pickle.dump(bst, fd)

        with open(filename, 'rb') as fd:
            bst = pickle.load(fd)

        assert bst.get_dump(dump_format='json') == dump_0

        if os.path.exists(filename):
            os.remove(filename)

    def test_model_pickling_binary(self):
        params = {
            'nthread': 1,
            'tree_method': 'hist'
        }
        self.run_model_pickling(params)

    def test_model_pickling_json(self):
        params = {
            'nthread': 1,
            'tree_method': 'hist',
            'enable_experimental_json_serialization': True
        }
        self.run_model_pickling(params)
