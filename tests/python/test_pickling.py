import json
import os
import pickle
import tempfile

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

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

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_with_sklearn_obj_metric(self) -> None:
        from sklearn.metrics import mean_squared_error

        X, y = tm.datasets.make_regression()
        reg = xgb.XGBRegressor(objective=tm.ls_obj, eval_metric=mean_squared_error)
        reg.fit(X, y)

        pkl = pickle.dumps(reg)
        reg_1 = pickle.loads(pkl)
        assert callable(reg_1.objective)
        assert callable(reg_1.eval_metric)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            reg.save_model(path)

            reg_2 = xgb.XGBRegressor()
            reg_2.load_model(path)

        assert not callable(reg_2.objective)
        assert not callable(reg_2.eval_metric)
        assert reg_2.eval_metric is None
