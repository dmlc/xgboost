import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.updater import get_basescore

rng = np.random.RandomState(1994)


class TestEarlyStopping:
    @pytest.mark.skipif(**tm.no_sklearn())
    def test_early_stopping_nonparallel(self):
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        digits = load_digits(n_class=2)
        X = digits['data']
        y = digits['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        clf1 = xgb.XGBClassifier(learning_rate=0.1)
        clf1.fit(X_train, y_train, early_stopping_rounds=5, eval_metric="auc",
                 eval_set=[(X_test, y_test)])
        clf2 = xgb.XGBClassifier(learning_rate=0.1)
        clf2.fit(X_train, y_train, early_stopping_rounds=4, eval_metric="auc",
                 eval_set=[(X_test, y_test)])
        # should be the same
        assert clf1.best_score == clf2.best_score
        assert clf1.best_score != 1
        # check overfit
        clf3 = xgb.XGBClassifier(
            learning_rate=0.1,
            eval_metric="auc",
            early_stopping_rounds=10
        )
        clf3.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        base_score = get_basescore(clf3)
        assert 0.53 > base_score > 0.5

        clf3 = xgb.XGBClassifier(
            learning_rate=0.1,
            base_score=.5,
            eval_metric="auc",
            early_stopping_rounds=10
        )
        clf3.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        assert clf3.best_score == 1

    def evalerror(self, preds, dtrain):
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))
        return 'rmse', mean_squared_error(labels, preds)

    @staticmethod
    def assert_metrics_length(cv, expected_length):
        for key, value in cv.items():
            assert len(value) == expected_length

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_cv_early_stopping(self):
        from sklearn.datasets import load_digits

        digits = load_digits(n_class=2)
        X = digits['data']
        y = digits['target']
        dm = xgb.DMatrix(X, label=y)
        params = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
                  'objective': 'binary:logistic', 'eval_metric': 'error'}

        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    early_stopping_rounds=10)
        self.assert_metrics_length(cv, 10)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    early_stopping_rounds=5)
        self.assert_metrics_length(cv, 3)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    early_stopping_rounds=1)
        self.assert_metrics_length(cv, 1)

        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    feval=self.evalerror, early_stopping_rounds=10)
        self.assert_metrics_length(cv, 10)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    feval=self.evalerror, early_stopping_rounds=1)
        self.assert_metrics_length(cv, 5)
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10,
                    feval=self.evalerror, maximize=True,
                    early_stopping_rounds=1)
        self.assert_metrics_length(cv, 1)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.skipif(**tm.no_pandas())
    def test_cv_early_stopping_with_multiple_eval_sets_and_metrics(self):
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        dm = xgb.DMatrix(X, label=y)
        params = {'objective':'binary:logistic'}

        metrics = [['auc'], ['error'], ['logloss'],
                   ['logloss', 'auc'], ['logloss', 'error'], ['error', 'logloss']]

        num_iteration_history = []

        # If more than one metrics is given, early stopping should use the last metric
        for i, m in enumerate(metrics):
            result = xgb.cv(params, dm, num_boost_round=1000, nfold=5, stratified=True,
                            metrics=m, early_stopping_rounds=20, seed=42)
            num_iteration_history.append(len(result))
            df = result['test-{}-mean'.format(m[-1])]
            # When early stopping is invoked, the last metric should be as best it can be.
            if m[-1] == 'auc':
                assert np.all(df <= df.iloc[-1])
            else:
                assert np.all(df >= df.iloc[-1])
        assert num_iteration_history[:3] == num_iteration_history[3:]
