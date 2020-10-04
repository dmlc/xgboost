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

    def test_early_stopping_customize(self):
        D_train = xgb.DMatrix(self.X_train, self.y_train)
        D_valid = xgb.DMatrix(self.X_valid, self.y_valid)
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                metric_name='PyError',
                                                data_name='Train')
        # Specify which dataset and which metric should be used for early stopping.
        booster = xgb.train(
            {'objective': 'binary:logistic',
             'eval_metric': ['error', 'rmse'],
             'tree_method': 'hist'}, D_train,
            evals=[(D_train, 'Train'), (D_valid, 'Valid')],
            feval=tm.eval_error_metric,
            num_boost_round=1000,
            callbacks=[early_stop],
            verbose_eval=False)
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1
        assert len(early_stop.stopping_history['Train']['PyError']) == len(dump)

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

    def run_eta_decay(self, tree_method, deprecated_callback):
        if deprecated_callback:
            scheduler = xgb.callback.reset_learning_rate
        else:
            scheduler = xgb.callback.LearningRateScheduler

        dpath = os.path.join(tm.PROJECT_ROOT, 'demo/data/')
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 4

        # learning_rates as a list
        # init eta with 0 to check whether learning_rates work
        param = {'max_depth': 2, 'eta': 0, 'verbosity': 0,
                 'objective': 'binary:logistic', 'eval_metric': 'error',
                 'tree_method': tree_method}
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[scheduler([
                            0.8, 0.7, 0.6, 0.5
                        ])],
                        evals_result=evals_result)
        eval_errors_0 = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should decrease, if eta > 0
        assert eval_errors_0[0] > eval_errors_0[-1]

        # init learning_rate with 0 to check whether learning_rates work
        param = {'max_depth': 2, 'learning_rate': 0, 'verbosity': 0,
                 'objective': 'binary:logistic', 'eval_metric': 'error',
                 'tree_method': tree_method}
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[scheduler(
                            [0.8, 0.7, 0.6, 0.5])],
                        evals_result=evals_result)
        eval_errors_1 = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should decrease, if learning_rate > 0
        assert eval_errors_1[0] > eval_errors_1[-1]

        # check if learning_rates override default value of eta/learning_rate
        param = {
            'max_depth': 2, 'verbosity': 0, 'objective': 'binary:logistic',
            'eval_metric': 'error', 'tree_method': tree_method
        }
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[scheduler(
                            [0, 0, 0, 0]
                        )],
                        evals_result=evals_result)
        eval_errors_2 = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should not decrease, if eta/learning_rate = 0
        assert eval_errors_2[0] == eval_errors_2[-1]

        # learning_rates as a customized decay function
        def eta_decay(ithround, num_boost_round=num_round):
            return num_boost_round / (ithround + 1)

        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        callbacks=[
                            scheduler(eta_decay)
                        ],
                        evals_result=evals_result)
        eval_errors_3 = list(map(float, evals_result['eval']['error']))

        assert isinstance(bst, xgb.core.Booster)

        assert eval_errors_3[0] == eval_errors_2[0]

        for i in range(1, len(eval_errors_0)):
            assert eval_errors_3[i] != eval_errors_2[i]

    def test_eta_decay_hist(self):
        # self.run_eta_decay('hist', True)
        self.run_eta_decay('hist', False)

    def test_eta_decay_approx(self):
        # self.run_eta_decay('approx', True)
        self.run_eta_decay('approx', False)

    def test_eta_decay_exact(self):
        # self.run_eta_decay('exact', True)
        self.run_eta_decay('exact', False)

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
