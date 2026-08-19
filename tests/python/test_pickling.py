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

        dump_0 = bst.get_dump(dump_format="json")
        assert dump_0
        config_0 = bst.save_config()

        filename = "model.pkl"

        with open(filename, "wb") as fd:
            pickle.dump(bst, fd)

        with open(filename, "rb") as fd:
            bst = pickle.load(fd)

        with open(filename, "wb") as fd:
            pickle.dump(bst, fd)

        with open(filename, "rb") as fd:
            bst = pickle.load(fd)

        assert bst.get_dump(dump_format="json") == dump_0

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

    def test_rng_state_is_portable(self):
        """The pickled RNG state must not depend on the C++ standard library that wrote
        it.

        Serializing the text form of ``std::mt19937`` made a pickle produced on Linux
        unreadable on Windows, since libstdc++, libc++ and the MSVC STL each spell that
        text differently.  See https://github.com/dmlc/xgboost/issues/12459 .
        """
        X, y = generate_data()
        params = {"tree_method": "exact", "subsample": 0.5, "seed": 1994}
        bst = xgb.train(params, xgb.DMatrix(X, y), num_boost_round=4)

        config = json.loads(bst.save_config())
        state = config["learner"]["generic_param"]["rng_state"]

        # The seed followed by a draw count, and nothing else.  The old format ran to
        # several hundred state words.
        seed, n_advanced = state.split(" ")
        assert seed == "1994"
        assert int(n_advanced) > 0

    def test_rng_state_survives_pickling(self):
        """Training resumed from a pickle must match uninterrupted training."""
        X, y = generate_data()
        dtrain = xgb.DMatrix(X, y)
        params = {"tree_method": "exact", "subsample": 0.5, "seed": 1994}

        straight = xgb.train(params, dtrain, num_boost_round=8)

        half = xgb.train(params, dtrain, num_boost_round=4)
        resumed = xgb.train(
            params,
            dtrain,
            num_boost_round=4,
            xgb_model=pickle.loads(pickle.dumps(half)),
        )

        np.testing.assert_allclose(
            straight.predict(dtrain), resumed.predict(dtrain), rtol=1e-6
        )
