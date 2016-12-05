import xgboost as xgb
import testing as tm
import numpy as np
import unittest

rng = np.random.RandomState(1337)


class TestTrainingContinuation(unittest.TestCase):
    num_parallel_tree = 3

    xgb_params_01 = {
        'silent': 1,
        'nthread': 1,
    }

    xgb_params_02 = {
        'silent': 1,
        'nthread': 1,
        'num_parallel_tree': num_parallel_tree
    }

    xgb_params_03 = {
        'silent': 1,
        'nthread': 1,
        'num_class': 5,
        'num_parallel_tree': num_parallel_tree
    }

    def test_training_continuation(self):
        tm._skip_if_no_sklearn()
        from sklearn.datasets import load_digits
        from sklearn.metrics import mean_squared_error

        digits_2class = load_digits(2)
        digits_5class = load_digits(5)

        X_2class = digits_2class['data']
        y_2class = digits_2class['target']

        X_5class = digits_5class['data']
        y_5class = digits_5class['target']

        dtrain_2class = xgb.DMatrix(X_2class, label=y_2class)
        dtrain_5class = xgb.DMatrix(X_5class, label=y_5class)

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain_2class, num_boost_round=10)
        ntrees_01 = len(gbdt_01.get_dump())
        assert ntrees_01 == 10

        gbdt_02 = xgb.train(self.xgb_params_01, dtrain_2class, num_boost_round=0)
        gbdt_02.save_model('xgb_tc.model')

        gbdt_02a = xgb.train(self.xgb_params_01, dtrain_2class, num_boost_round=10, xgb_model=gbdt_02)
        gbdt_02b = xgb.train(self.xgb_params_01, dtrain_2class, num_boost_round=10, xgb_model="xgb_tc.model")
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

        gbdt_03 = xgb.train(self.xgb_params_01, dtrain_2class, num_boost_round=3)
        gbdt_03.save_model('xgb_tc.model')

        gbdt_03a = xgb.train(self.xgb_params_01, dtrain_2class, num_boost_round=7, xgb_model=gbdt_03)
        gbdt_03b = xgb.train(self.xgb_params_01, dtrain_2class, num_boost_round=7, xgb_model="xgb_tc.model")
        ntrees_03a = len(gbdt_03a.get_dump())
        ntrees_03b = len(gbdt_03b.get_dump())
        assert ntrees_03a == 10
        assert ntrees_03b == 10

        res1 = mean_squared_error(y_2class, gbdt_03a.predict(dtrain_2class))
        res2 = mean_squared_error(y_2class, gbdt_03b.predict(dtrain_2class))
        assert res1 == res2

        gbdt_04 = xgb.train(self.xgb_params_02, dtrain_2class, num_boost_round=3)
        assert gbdt_04.best_ntree_limit == (gbdt_04.best_iteration + 1) * self.num_parallel_tree

        res1 = mean_squared_error(y_2class, gbdt_04.predict(dtrain_2class))
        res2 = mean_squared_error(y_2class, gbdt_04.predict(dtrain_2class, ntree_limit=gbdt_04.best_ntree_limit))
        assert res1 == res2

        gbdt_04 = xgb.train(self.xgb_params_02, dtrain_2class, num_boost_round=7, xgb_model=gbdt_04)
        assert gbdt_04.best_ntree_limit == (gbdt_04.best_iteration + 1) * self.num_parallel_tree

        res1 = mean_squared_error(y_2class, gbdt_04.predict(dtrain_2class))
        res2 = mean_squared_error(y_2class, gbdt_04.predict(dtrain_2class, ntree_limit=gbdt_04.best_ntree_limit))
        assert res1 == res2

        gbdt_05 = xgb.train(self.xgb_params_03, dtrain_5class, num_boost_round=7)
        assert gbdt_05.best_ntree_limit == (gbdt_05.best_iteration + 1) * self.num_parallel_tree
        gbdt_05 = xgb.train(self.xgb_params_03, dtrain_5class, num_boost_round=3, xgb_model=gbdt_05)
        assert gbdt_05.best_ntree_limit == (gbdt_05.best_iteration + 1) * self.num_parallel_tree

        res1 = gbdt_05.predict(dtrain_5class)
        res2 = gbdt_05.predict(dtrain_5class, ntree_limit=gbdt_05.best_ntree_limit)
        np.testing.assert_almost_equal(res1, res2)
