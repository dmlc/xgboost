import xgboost as xgb
import unittest
import pytest
import os
import testing as tm
import tempfile

# We use the dataset for tests.
pytestmark = pytest.mark.skipif(**tm.no_sklearn())


class TestCallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        cls.X = X
        cls.y = y

        split = int(X.shape[0]*0.8)
        cls.X_train = X[: split, ...]
        cls.y_train = y[: split, ...]
        cls.X_valid = X[split:, ...]
        cls.y_valid = y[split:, ...]

    def test_evaluation_monitor(self):
        D_train = xgb.DMatrix(self.X_train, self.y_train)
        D_valid = xgb.DMatrix(self.X_valid, self.y_valid)
        evals_result = {}
        rounds = 10
        xgb.train({'objective': 'binary:logistic'}, D_train,
                  evals=[(D_train, 'Train'), (D_valid, 'Valid')],
                  num_boost_round=rounds,
                  evals_result=evals_result,
                  verbose_eval=True)
        print('evals_result:', evals_result)
        assert len(evals_result['Train']['error']) == rounds
        assert len(evals_result['Valid']['error']) == rounds

    def test_early_stopping(self):
        D_train = xgb.DMatrix(self.X_train, self.y_train)
        D_valid = xgb.DMatrix(self.X_valid, self.y_valid)
        evals_result = {}
        rounds = 30
        early_stopping_rounds = 5
        booster = xgb.train({'objective': 'binary:logistic'}, D_train,
                            evals=[(D_train, 'Train'), (D_valid, 'Valid')],
                            num_boost_round=rounds,
                            evals_result=evals_result,
                            verbose_eval=True,
                            early_stopping_rounds=early_stopping_rounds)
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_custom_eval(self):
        D_train = xgb.DMatrix(self.X_train, self.y_train)
        D_valid = xgb.DMatrix(self.X_valid, self.y_valid)
        early_stopping_rounds = 5
        booster = xgb.train({'objective': 'binary:logistic',
                             'tree_method': 'hist'}, D_train,
                            evals=[(D_train, 'Train'), (D_valid, 'Valid')],
                            feval=tm.eval_error_metric,
                            num_boost_round=1000,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=False)
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_skl(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        cls = xgb.XGBClassifier()
        early_stopping_rounds = 5
        cls.fit(X, y, eval_set=[(X, y)],
                early_stopping_rounds=early_stopping_rounds)
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_custom_eval_skl(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        cls = xgb.XGBClassifier()
        early_stopping_rounds = 5
        cls.fit(X, y, eval_set=[(X, y)],
                early_stopping_rounds=early_stopping_rounds)
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_check_point(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        m = xgb.DMatrix(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                          rounds=1,
                                                          name='model')
            xgb.train({'objective': 'binary:logistic'}, m,
                      num_boost_round=10,
                      verbose_eval=False,
                      callbacks=[check_point])
            for i in range(0, 10):
                assert os.path.exists(
                    os.path.join(tmpdir, 'model_' + str(i) + '.json'))

            check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                          rounds=1,
                                                          as_pickle=True,
                                                          name='model')
            xgb.train({'objective': 'binary:logistic'}, m,
                      num_boost_round=10,
                      verbose_eval=False,
                      callbacks=[check_point])
            for i in range(0, 10):
                assert os.path.exists(
                    os.path.join(tmpdir, 'model_' + str(i) + '.pkl'))

# def test_learning_rate_scheduler():
#     pass
