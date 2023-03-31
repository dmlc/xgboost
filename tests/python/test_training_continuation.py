import os
import tempfile

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm

rng = np.random.RandomState(1337)


class TestTrainingContinuation:
    num_parallel_tree = 3

    def generate_parameters(self):
        xgb_params_01_binary = {
            'nthread': 1,
        }

        xgb_params_02_binary = {
            'nthread': 1,
            'num_parallel_tree': self.num_parallel_tree
        }

        xgb_params_03_binary = {
            'nthread': 1,
            'num_class': 5,
            'num_parallel_tree': self.num_parallel_tree
        }

        return [
            xgb_params_01_binary, xgb_params_02_binary, xgb_params_03_binary
        ]

    def run_training_continuation(self, xgb_params_01, xgb_params_02,
                                  xgb_params_03):
        from sklearn.datasets import load_digits
        from sklearn.metrics import mean_squared_error

        digits_2class = load_digits(n_class=2)
        digits_5class = load_digits(n_class=5)

        X_2class = digits_2class['data']
        y_2class = digits_2class['target']

        X_5class = digits_5class['data']
        y_5class = digits_5class['target']

        dtrain_2class = xgb.DMatrix(X_2class, label=y_2class)
        dtrain_5class = xgb.DMatrix(X_5class, label=y_5class)

        gbdt_01 = xgb.train(xgb_params_01, dtrain_2class,
                            num_boost_round=10)
        ntrees_01 = len(gbdt_01.get_dump())
        assert ntrees_01 == 10

        gbdt_02 = xgb.train(xgb_params_01, dtrain_2class,
                            num_boost_round=0)
        gbdt_02.save_model('xgb_tc.model')

        gbdt_02a = xgb.train(xgb_params_01, dtrain_2class,
                             num_boost_round=10, xgb_model=gbdt_02)
        gbdt_02b = xgb.train(xgb_params_01, dtrain_2class,
                             num_boost_round=10, xgb_model="xgb_tc.model")
        ntrees_02a = len(gbdt_02a.get_dump())
        ntrees_02b = len(gbdt_02b.get_dump())
        assert ntrees_02a == 10
        assert ntrees_02b == 10

        res1 = mean_squared_error(y_2class, gbdt_01.predict(dtrain_2class))
        res2 = mean_squared_error(y_2class, gbdt_02a.predict(dtrain_2class))
        assert res1 == res2

        res1 = mean_squared_error(y_2class, gbdt_01.predict(dtrain_2class))
        res2 = mean_squared_error(y_2class, gbdt_02b.predict(dtrain_2class))
        assert res1 == res2

        gbdt_03 = xgb.train(xgb_params_01, dtrain_2class,
                            num_boost_round=3)
        gbdt_03.save_model('xgb_tc.model')

        gbdt_03a = xgb.train(xgb_params_01, dtrain_2class,
                             num_boost_round=7, xgb_model=gbdt_03)
        gbdt_03b = xgb.train(xgb_params_01, dtrain_2class,
                             num_boost_round=7, xgb_model="xgb_tc.model")
        ntrees_03a = len(gbdt_03a.get_dump())
        ntrees_03b = len(gbdt_03b.get_dump())
        assert ntrees_03a == 10
        assert ntrees_03b == 10

        os.remove('xgb_tc.model')

        res1 = mean_squared_error(y_2class, gbdt_03a.predict(dtrain_2class))
        res2 = mean_squared_error(y_2class, gbdt_03b.predict(dtrain_2class))
        assert res1 == res2

        gbdt_04 = xgb.train(xgb_params_02, dtrain_2class, num_boost_round=3)
        res1 = mean_squared_error(y_2class, gbdt_04.predict(dtrain_2class))
        res2 = mean_squared_error(
            y_2class,
            gbdt_04.predict(
                dtrain_2class, iteration_range=(0, gbdt_04.best_iteration + 1)
            )
        )
        assert res1 == res2

        gbdt_04 = xgb.train(
            xgb_params_02, dtrain_2class, num_boost_round=7, xgb_model=gbdt_04
        )
        res1 = mean_squared_error(y_2class, gbdt_04.predict(dtrain_2class))
        res2 = mean_squared_error(
            y_2class,
            gbdt_04.predict(
                dtrain_2class, iteration_range=(0, gbdt_04.best_iteration + 1)
            )
        )
        assert res1 == res2

        gbdt_05 = xgb.train(xgb_params_03, dtrain_5class,
                            num_boost_round=7)
        gbdt_05 = xgb.train(xgb_params_03,
                            dtrain_5class,
                            num_boost_round=3,
                            xgb_model=gbdt_05)

        res1 = gbdt_05.predict(dtrain_5class)
        res2 = gbdt_05.predict(
            dtrain_5class, iteration_range=(0, gbdt_05.best_iteration + 1)
        )
        np.testing.assert_almost_equal(res1, res2)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_training_continuation_json(self):
        params = self.generate_parameters()
        self.run_training_continuation(params[0], params[1], params[2])

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_training_continuation_updaters_json(self):
        # Picked up from R tests.
        updaters = 'grow_colmaker,prune,refresh'
        params = self.generate_parameters()
        for p in params:
            p['updater'] = updaters
        self.run_training_continuation(params[0], params[1], params[2])

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_changed_parameter(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        clf = xgb.XGBClassifier(n_estimators=2)
        clf.fit(X, y, eval_set=[(X, y)], eval_metric="logloss")
        assert tm.non_increasing(clf.evals_result()["validation_0"]["logloss"])

        with tempfile.TemporaryDirectory() as tmpdir:
            clf.save_model(os.path.join(tmpdir, "clf.json"))
            loaded = xgb.XGBClassifier()
            loaded.load_model(os.path.join(tmpdir, "clf.json"))

        clf = xgb.XGBClassifier(n_estimators=2)
        # change metric to error
        clf.fit(X, y, eval_set=[(X, y)], eval_metric="error")
        assert tm.non_increasing(clf.evals_result()["validation_0"]["error"])
