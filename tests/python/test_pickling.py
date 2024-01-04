import json
import os
import pickle

import numpy as np

import xgboost as xgb

kRows = 100
kCols = 10


def generate_data():
    X = np.random.randn(kRows, kCols)
    y = np.random.randn(kRows)
    return X, y


class TestPickling:
    def run_model_pickling(self, xgb_params) -> str:
        X, y = generate_data()
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(xgb_params, dtrain)

        dump_0 = bst.get_dump(dump_format='json')
        assert dump_0
        config_0 = bst.save_config()

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

        config_1 = bst.save_config()
        assert config_0 == config_1
        return json.loads(config_0)

    def test_model_pickling_json(self):
        def check(config):
            tree_param = config["learner"]["gradient_booster"]["tree_train_param"]
            subsample = tree_param["subsample"]
            assert float(subsample) == 0.5

        params = {"nthread": 8, "tree_method": "hist", "subsample": 0.5}
        config = self.run_model_pickling(params)
        check(config)
        params = {"nthread": 8, "tree_method": "exact", "subsample": 0.5}
        config = self.run_model_pickling(params)
        check(config)
