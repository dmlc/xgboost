import xgboost as xgb
import numpy as np
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston
import unittest

rng = np.random.RandomState(1337)

class TestTrainingContinuation(unittest.TestCase):

    xgb_params = {
        'colsample_bytree': 0.7,
        'silent': 1,
        'nthread': 1,
    }

    def test_training_continuation(self):
        digits = load_digits(2)
        X = digits['data']
        y = digits['target']

        dtrain = xgb.DMatrix(X,label=y)

        gbdt_01 = xgb.train(self.xgb_params, dtrain, num_boost_round=10)
        ntrees_01 = len(gbdt_01.get_dump())
        assert ntrees_01 == 10

        gbdt_02 = xgb.train(self.xgb_params, dtrain, num_boost_round=0)
        gbdt_02.save_model('xgb_tc.model')

        gbdt_02a = xgb.train(self.xgb_params, dtrain, num_boost_round=10, xgb_model=gbdt_02)
        gbdt_02b = xgb.train(self.xgb_params, dtrain, num_boost_round=10, xgb_model="xgb_tc.model")
        ntrees_02a = len(gbdt_02a.get_dump())
        ntrees_02b = len(gbdt_02b.get_dump())
        assert ntrees_02a == 10
        assert ntrees_02b == 10
        assert mean_squared_error(y, gbdt_01.predict(dtrain)) == mean_squared_error(y, gbdt_02a.predict(dtrain))
        assert mean_squared_error(y, gbdt_01.predict(dtrain)) == mean_squared_error(y, gbdt_02b.predict(dtrain))

        gbdt_03 = xgb.train(self.xgb_params, dtrain, num_boost_round=3)
        gbdt_03.save_model('xgb_tc.model')

        gbdt_03a = xgb.train(self.xgb_params, dtrain, num_boost_round=7, xgb_model=gbdt_03)
        gbdt_03b = xgb.train(self.xgb_params, dtrain, num_boost_round=7, xgb_model="xgb_tc.model")
        ntrees_03a = len(gbdt_03a.get_dump())
        ntrees_03b = len(gbdt_03b.get_dump())
        assert ntrees_03a == 10
        assert ntrees_03b == 10
        assert mean_squared_error(y, gbdt_03a.predict(dtrain)) == mean_squared_error(y, gbdt_03b.predict(dtrain))
		
