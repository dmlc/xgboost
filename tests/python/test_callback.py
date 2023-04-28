import json
import os
import tempfile
from contextlib import nullcontext
from typing import Union

import pytest

import xgboost as xgb
from xgboost import testing as tm

# We use the dataset for tests.
pytestmark = pytest.mark.skipif(**tm.no_sklearn())


class TestCallbacks:
    @classmethod
    def setup_class(cls):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        cls.X = X
        cls.y = y

        split = int(X.shape[0]*0.8)
        cls.X_train = X[: split, ...]
        cls.y_train = y[: split, ...]
        cls.X_valid = X[split:, ...]
        cls.y_valid = y[split:, ...]

    def run_evaluation_monitor(
        self,
        D_train: xgb.DMatrix,
        D_valid: xgb.DMatrix,
        rounds: int,
        verbose_eval: Union[bool, int]
    ):
        def check_output(output: str) -> None:
            if int(verbose_eval) == 1:
                # Should print each iteration info
                assert len(output.split('\n')) == rounds
            elif int(verbose_eval) > rounds:
                # Should print first and latest iteration info
                assert len(output.split('\n')) == 2
            else:
                # Should print info by each period additionaly to first and latest
                # iteration
                num_periods = rounds // int(verbose_eval)
                # Extra information is required for latest iteration
                is_extra_info_required = num_periods * int(verbose_eval) < (rounds - 1)
                assert len(output.split('\n')) == (
                    1 + num_periods + int(is_extra_info_required)
                )

        evals_result: xgb.callback.TrainingCallback.EvalsLog = {}
        params = {'objective': 'binary:logistic', 'eval_metric': 'error'}
        with tm.captured_output() as (out, err):
            xgb.train(
                params, D_train,
                evals=[(D_train, 'Train'), (D_valid, 'Valid')],
                num_boost_round=rounds,
                evals_result=evals_result,
                verbose_eval=verbose_eval,
            )
            output: str = out.getvalue().strip()
            check_output(output)

        with tm.captured_output() as (out, err):
            xgb.cv(params, D_train, num_boost_round=rounds, verbose_eval=verbose_eval)
            output = out.getvalue().strip()
            check_output(output)

    def test_evaluation_monitor(self):
        D_train = xgb.DMatrix(self.X_train, self.y_train)
        D_valid = xgb.DMatrix(self.X_valid, self.y_valid)
        evals_result = {}
        rounds = 10
        xgb.train({'objective': 'binary:logistic',
                   'eval_metric': 'error'}, D_train,
                  evals=[(D_train, 'Train'), (D_valid, 'Valid')],
                  num_boost_round=rounds,
                  evals_result=evals_result,
                  verbose_eval=True)
        assert len(evals_result['Train']['error']) == rounds
        assert len(evals_result['Valid']['error']) == rounds

        self.run_evaluation_monitor(D_train, D_valid, rounds, True)
        self.run_evaluation_monitor(D_train, D_valid, rounds, 2)
        self.run_evaluation_monitor(D_train, D_valid, rounds, 4)
        self.run_evaluation_monitor(D_train, D_valid, rounds, rounds + 1)

    def test_early_stopping(self):
        D_train = xgb.DMatrix(self.X_train, self.y_train)
        D_valid = xgb.DMatrix(self.X_valid, self.y_valid)
        evals_result = {}
        rounds = 30
        early_stopping_rounds = 5
        booster = xgb.train({'objective': 'binary:logistic',
                             'eval_metric': 'error'}, D_train,
                            evals=[(D_train, 'Train'), (D_valid, 'Valid')],
                            num_boost_round=rounds,
                            evals_result=evals_result,
                            verbose_eval=True,
                            early_stopping_rounds=early_stopping_rounds)
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

        # No early stopping, best_iteration should be set to last epoch
        booster = xgb.train({'objective': 'binary:logistic',
                             'eval_metric': 'error'}, D_train,
                            evals=[(D_train, 'Train'), (D_valid, 'Valid')],
                            num_boost_round=10,
                            evals_result=evals_result,
                            verbose_eval=True)
        assert booster.num_boosted_rounds() - 1 == booster.best_iteration

    def test_early_stopping_custom_eval(self):
        D_train = xgb.DMatrix(self.X_train, self.y_train)
        D_valid = xgb.DMatrix(self.X_valid, self.y_valid)
        early_stopping_rounds = 5
        booster = xgb.train({'objective': 'binary:logistic',
                             'eval_metric': 'error',
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
                                                metric_name='CustomErr',
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
        assert len(early_stop.stopping_history['Train']['CustomErr']) == len(dump)

        rounds = 100
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name='CustomErr',
            data_name='Train',
            min_delta=100,
            save_best=True,
        )
        booster = xgb.train(
            {
                'objective': 'binary:logistic',
                'eval_metric': ['error', 'rmse'],
                'tree_method': 'hist'
            },
            D_train,
            evals=[(D_train, 'Train'), (D_valid, 'Valid')],
            feval=tm.eval_error_metric,
            num_boost_round=rounds,
            callbacks=[early_stop],
            verbose_eval=False
        )
        # No iteration can be made with min_delta == 100
        assert booster.best_iteration == 0
        assert booster.num_boosted_rounds() == 1

    def test_early_stopping_skl(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        early_stopping_rounds = 5
        cls = xgb.XGBClassifier(
            early_stopping_rounds=early_stopping_rounds, eval_metric='error'
        )
        cls.fit(X, y, eval_set=[(X, y)])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_custom_eval_skl(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds)
        cls = xgb.XGBClassifier(
            eval_metric=tm.eval_error_metric_skl, callbacks=[early_stop]
        )
        cls.fit(X, y, eval_set=[(X, y)])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_save_best_model(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        n_estimators = 100
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                save_best=True)
        cls = xgb.XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=tm.eval_error_metric_skl,
            callbacks=[early_stop]
        )
        cls.fit(X, y, eval_set=[(X, y)])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format='json')
        assert len(dump) == booster.best_iteration + 1

        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                save_best=True)
        cls = xgb.XGBClassifier(
            booster='gblinear', n_estimators=10, eval_metric=tm.eval_error_metric_skl
        )
        with pytest.raises(ValueError):
            cls.fit(X, y, eval_set=[(X, y)], callbacks=[early_stop])

        # No error
        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                save_best=False)
        xgb.XGBClassifier(
            booster='gblinear', n_estimators=10, eval_metric=tm.eval_error_metric_skl
        ).fit(X, y, eval_set=[(X, y)], callbacks=[early_stop])

    def test_early_stopping_continuation(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        cls = xgb.XGBClassifier(eval_metric=tm.eval_error_metric_skl)
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, save_best=True
        )
        with pytest.warns(UserWarning):
            cls.fit(X, y, eval_set=[(X, y)], callbacks=[early_stop])

        booster = cls.get_booster()
        assert booster.num_boosted_rounds() == booster.best_iteration + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.json')
            cls.save_model(path)
            cls = xgb.XGBClassifier()
            cls.load_model(path)
            assert cls._Booster is not None
            early_stopping_rounds = 3
            cls.set_params(eval_metric=tm.eval_error_metric_skl)
            cls.fit(X, y, eval_set=[(X, y)], early_stopping_rounds=early_stopping_rounds)
            booster = cls.get_booster()
            assert booster.num_boosted_rounds() == \
                booster.best_iteration + early_stopping_rounds + 1

    def test_deprecated(self):
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, save_best=True
        )
        clf = xgb.XGBClassifier(
            eval_metric=tm.eval_error_metric_skl, callbacks=[early_stop]
        )
        with pytest.raises(ValueError, match=r".*set_params.*"):
            clf.fit(X, y, eval_set=[(X, y)], callbacks=[early_stop])

    def run_eta_decay(self, tree_method):
        """Test learning rate scheduler, used by both CPU and GPU tests."""
        scheduler = xgb.callback.LearningRateScheduler

        dtrain, dtest = tm.load_agaricus(__file__)

        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 4

        warning_check = nullcontext()

        # learning_rates as a list
        # init eta with 0 to check whether learning_rates work
        param = {'max_depth': 2, 'eta': 0, 'verbosity': 0,
                 'objective': 'binary:logistic', 'eval_metric': 'error',
                 'tree_method': tree_method}
        evals_result = {}
        with warning_check:
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
        with warning_check:
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
        with warning_check:
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
        with warning_check:
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

        with warning_check:
            xgb.cv(param, dtrain, num_round, callbacks=[scheduler(eta_decay)])

    def run_eta_decay_leaf_output(self, tree_method: str, objective: str) -> None:
        # check decay has effect on leaf output.
        num_round = 4
        scheduler = xgb.callback.LearningRateScheduler

        dtrain, dtest = tm.load_agaricus(__file__)
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]

        param = {
            "max_depth": 2,
            "objective": objective,
            "eval_metric": "error",
            "tree_method": tree_method,
        }
        if objective == "reg:quantileerror":
            param["quantile_alpha"] = 0.3

        def eta_decay_0(i):
            return num_round / (i + 1)

        bst0 = xgb.train(
            param,
            dtrain,
            num_round,
            watchlist,
            callbacks=[scheduler(eta_decay_0)],
        )

        def eta_decay_1(i: int) -> float:
            if i > 1:
                return 5.0
            return num_round / (i + 1)

        bst1 = xgb.train(
            param,
            dtrain,
            num_round,
            watchlist,
            callbacks=[scheduler(eta_decay_1)],
        )
        bst_json0 = bst0.save_raw(raw_format="json")
        bst_json1 = bst1.save_raw(raw_format="json")

        j0 = json.loads(bst_json0)
        j1 = json.loads(bst_json1)

        tree_2th_0 = j0["learner"]["gradient_booster"]["model"]["trees"][2]
        tree_2th_1 = j1["learner"]["gradient_booster"]["model"]["trees"][2]
        assert tree_2th_0["base_weights"] == tree_2th_1["base_weights"]
        assert tree_2th_0["split_conditions"] == tree_2th_1["split_conditions"]

        tree_3th_0 = j0["learner"]["gradient_booster"]["model"]["trees"][3]
        tree_3th_1 = j1["learner"]["gradient_booster"]["model"]["trees"][3]
        assert tree_3th_0["base_weights"] != tree_3th_1["base_weights"]
        assert tree_3th_0["split_conditions"] != tree_3th_1["split_conditions"]

    @pytest.mark.parametrize("tree_method", ["hist", "approx", "approx"])
    def test_eta_decay(self, tree_method):
        self.run_eta_decay(tree_method)

    @pytest.mark.parametrize(
        "tree_method,objective",
        [
            ("hist", "binary:logistic"),
            ("hist", "reg:absoluteerror"),
            ("hist", "reg:quantileerror"),
            ("approx", "binary:logistic"),
            ("approx", "reg:absoluteerror"),
            ("approx", "reg:quantileerror"),
        ],
    )
    def test_eta_decay_leaf_output(self, tree_method: str, objective: str) -> None:
        self.run_eta_decay_leaf_output(tree_method, objective)

    def test_check_point(self):
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True)
        m = xgb.DMatrix(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            check_point = xgb.callback.TrainingCheckPoint(
                directory=tmpdir, iterations=1, name="model"
            )
            xgb.train(
                {"objective": "binary:logistic"},
                m,
                num_boost_round=10,
                verbose_eval=False,
                callbacks=[check_point],
            )
            for i in range(1, 10):
                assert os.path.exists(os.path.join(tmpdir, "model_" + str(i) + ".json"))

            check_point = xgb.callback.TrainingCheckPoint(
                directory=tmpdir, iterations=1, as_pickle=True, name="model"
            )
            xgb.train(
                {"objective": "binary:logistic"},
                m,
                num_boost_round=10,
                verbose_eval=False,
                callbacks=[check_point],
            )
            for i in range(1, 10):
                assert os.path.exists(os.path.join(tmpdir, "model_" + str(i) + ".pkl"))

    def test_callback_list(self):
        X, y = tm.data.get_california_housing()
        m = xgb.DMatrix(X, y)
        callbacks = [xgb.callback.EarlyStopping(rounds=10)]
        for i in range(4):
            xgb.train(
                {"objective": "reg:squarederror", "eval_metric": "rmse"},
                m,
                evals=[(m, "Train")],
                num_boost_round=1,
                verbose_eval=True,
                callbacks=callbacks,
            )
        assert len(callbacks) == 1
