import os
import tempfile
from collections import namedtuple
from typing import Tuple, Union

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.callbacks import (
    run_eta_decay,
    run_eta_decay_leaf_output,
    tree_methods_objs,
)

# We use the dataset for tests.
pytestmark = pytest.mark.skipif(**tm.no_sklearn())


BreastCancer = namedtuple("BreastCancer", ["full", "tr", "va"])


@pytest.fixture
def breast_cancer() -> BreastCancer:
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)

    split = int(X.shape[0] * 0.8)
    return BreastCancer(
        full=(X, y),
        tr=(X[:split, ...], y[:split, ...]),
        va=(X[split:, ...], y[split:, ...]),
    )


def eval_error_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, np.float64]:
    # No custom objective, recieve transformed output
    return tm.eval_error_metric(predt, dtrain, rev_link=False)


class TestCallbacks:
    def run_evaluation_monitor(
        self,
        D_train: xgb.DMatrix,
        D_valid: xgb.DMatrix,
        rounds: int,
        verbose_eval: Union[bool, int],
    ):
        def check_output(output: str) -> None:
            if int(verbose_eval) == 1:
                # Should print each iteration info
                assert len(output.split("\n")) == rounds
            elif int(verbose_eval) > rounds:
                # Should print first and latest iteration info
                assert len(output.split("\n")) == 2
            else:
                # Should print info by each period additionaly to first and latest
                # iteration
                num_periods = rounds // int(verbose_eval)
                # Extra information is required for latest iteration
                is_extra_info_required = num_periods * int(verbose_eval) < (rounds - 1)
                assert len(output.split("\n")) == (
                    1 + num_periods + int(is_extra_info_required)
                )

        evals_result: xgb.callback.TrainingCallback.EvalsLog = {}
        params = {"objective": "binary:logistic", "eval_metric": "error"}
        with tm.captured_output() as (out, err):
            xgb.train(
                params,
                D_train,
                evals=[(D_train, "Train"), (D_valid, "Valid")],
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

    def test_evaluation_monitor(self, breast_cancer: BreastCancer) -> None:
        D_train = xgb.DMatrix(breast_cancer.tr[0], breast_cancer.tr[1])
        D_valid = xgb.DMatrix(breast_cancer.va[0], breast_cancer.va[1])
        evals_result = {}
        rounds = 10
        xgb.train(
            {"objective": "binary:logistic", "eval_metric": "error"},
            D_train,
            evals=[(D_train, "Train"), (D_valid, "Valid")],
            num_boost_round=rounds,
            evals_result=evals_result,
            verbose_eval=True,
        )
        assert len(evals_result["Train"]["error"]) == rounds
        assert len(evals_result["Valid"]["error"]) == rounds

        self.run_evaluation_monitor(D_train, D_valid, rounds, True)
        self.run_evaluation_monitor(D_train, D_valid, rounds, 2)
        self.run_evaluation_monitor(D_train, D_valid, rounds, 4)
        self.run_evaluation_monitor(D_train, D_valid, rounds, rounds + 1)

    def test_early_stopping(self, breast_cancer: BreastCancer) -> None:
        D_train = xgb.DMatrix(breast_cancer.tr[0], breast_cancer.tr[1])
        D_valid = xgb.DMatrix(breast_cancer.va[0], breast_cancer.va[1])
        evals_result = {}
        rounds = 30
        early_stopping_rounds = 5
        booster = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "error"},
            D_train,
            evals=[(D_train, "Train"), (D_valid, "Valid")],
            num_boost_round=rounds,
            evals_result=evals_result,
            verbose_eval=True,
            early_stopping_rounds=early_stopping_rounds,
        )
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_custom_eval(self, breast_cancer: BreastCancer) -> None:
        D_train = xgb.DMatrix(breast_cancer.tr[0], breast_cancer.tr[1])
        D_valid = xgb.DMatrix(breast_cancer.va[0], breast_cancer.va[1])
        early_stopping_rounds = 5
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "error",
                "tree_method": "hist",
            },
            D_train,
            evals=[(D_train, "Train"), (D_valid, "Valid")],
            custom_metric=eval_error_metric,
            num_boost_round=1000,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_customize(self, breast_cancer: BreastCancer) -> None:
        D_train = xgb.DMatrix(breast_cancer.tr[0], breast_cancer.tr[1])
        D_valid = xgb.DMatrix(breast_cancer.va[0], breast_cancer.va[1])
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, metric_name="CustomErr", data_name="Train"
        )
        # Specify which dataset and which metric should be used for early stopping.
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": ["error", "rmse"],
                "tree_method": "hist",
            },
            D_train,
            evals=[(D_train, "Train"), (D_valid, "Valid")],
            custom_metric=eval_error_metric,
            num_boost_round=1000,
            callbacks=[early_stop],
            verbose_eval=False,
        )
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1
        assert len(early_stop.stopping_history["Train"]["CustomErr"]) == len(dump)

        rounds = 100
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="CustomErr",
            data_name="Train",
            min_delta=100,
            save_best=True,
        )
        booster = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": ["error", "rmse"],
                "tree_method": "hist",
            },
            D_train,
            evals=[(D_train, "Train"), (D_valid, "Valid")],
            # No custom objective, transformed output
            custom_metric=eval_error_metric,
            num_boost_round=rounds,
            callbacks=[early_stop],
            verbose_eval=False,
        )
        # No iteration can be made with min_delta == 100
        assert booster.best_iteration == 0
        assert booster.num_boosted_rounds() == 1

    def test_early_stopping_skl(self, breast_cancer: BreastCancer) -> None:
        X, y = breast_cancer.full
        early_stopping_rounds = 5
        cls = xgb.XGBClassifier(
            early_stopping_rounds=early_stopping_rounds, eval_metric="error"
        )
        cls.fit(X, y, eval_set=[(X, y)])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_custom_eval_skl(self, breast_cancer: BreastCancer) -> None:
        X, y = breast_cancer.full
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds)
        cls = xgb.XGBClassifier(
            eval_metric=tm.eval_error_metric_skl, callbacks=[early_stop]
        )
        cls.fit(X, y, eval_set=[(X, y)])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format="json")
        assert len(dump) - booster.best_iteration == early_stopping_rounds + 1

    def test_early_stopping_save_best_model(self, breast_cancer: BreastCancer) -> None:
        X, y = breast_cancer.full
        n_estimators = 100
        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, save_best=True
        )
        cls = xgb.XGBClassifier(
            n_estimators=n_estimators,
            eval_metric=tm.eval_error_metric_skl,
            callbacks=[early_stop],
        )
        cls.fit(X, y, eval_set=[(X, y)])
        booster = cls.get_booster()
        dump = booster.get_dump(dump_format="json")
        assert len(dump) == booster.best_iteration + 1

        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, save_best=True
        )
        cls = xgb.XGBClassifier(
            booster="gblinear",
            n_estimators=10,
            eval_metric=tm.eval_error_metric_skl,
            callbacks=[early_stop],
        )
        with pytest.raises(ValueError):
            cls.fit(X, y, eval_set=[(X, y)])

        # No error
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, save_best=False
        )
        xgb.XGBClassifier(
            booster="gblinear",
            n_estimators=10,
            eval_metric=tm.eval_error_metric_skl,
            callbacks=[early_stop],
        ).fit(X, y, eval_set=[(X, y)])

    def test_early_stopping_continuation(self, breast_cancer: BreastCancer) -> None:
        X, y = breast_cancer.full

        early_stopping_rounds = 5
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, save_best=True
        )
        cls = xgb.XGBClassifier(
            eval_metric=tm.eval_error_metric_skl, callbacks=[early_stop]
        )
        cls.fit(X, y, eval_set=[(X, y)])

        booster = cls.get_booster()
        assert booster.num_boosted_rounds() == booster.best_iteration + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            cls.save_model(path)
            cls = xgb.XGBClassifier()
            cls.load_model(path)
            assert cls._Booster is not None
            early_stopping_rounds = 3
            cls.set_params(
                eval_metric=tm.eval_error_metric_skl,
                early_stopping_rounds=early_stopping_rounds,
            )
            cls.fit(X, y, eval_set=[(X, y)])
            booster = cls.get_booster()
            assert (
                booster.num_boosted_rounds()
                == booster.best_iteration + early_stopping_rounds + 1
            )

    def test_early_stopping_multiple_metrics(self):
        from sklearn.datasets import make_classification

        X, y = make_classification(random_state=1994)
        # AUC approaches 1.0 real quick.
        clf = xgb.XGBClassifier(eval_metric=["logloss", "auc"], early_stopping_rounds=2)
        clf.fit(X, y, eval_set=[(X, y)])
        assert clf.best_iteration < 8
        assert clf.evals_result()["validation_0"]["auc"][-1] > 0.99

        clf = xgb.XGBClassifier(eval_metric=["auc", "logloss"], early_stopping_rounds=2)
        clf.fit(X, y, eval_set=[(X, y)])

        assert clf.best_iteration > 50
        assert clf.evals_result()["validation_0"]["auc"][-1] > 0.99

    @pytest.mark.parametrize("tree_method", ["hist", "approx", "exact"])
    def test_eta_decay(self, tree_method: str) -> None:
        dtrain, dtest = tm.load_agaricus(__file__)
        run_eta_decay(tree_method, dtrain, dtest, "cpu")

    @pytest.mark.parametrize("tree_method,objective", tree_methods_objs())
    def test_eta_decay_leaf_output(self, tree_method: str, objective: str) -> None:
        dtrain, dtest = tm.load_agaricus(__file__)
        run_eta_decay_leaf_output(tree_method, objective, dtrain, dtest, "cpu")

    def test_check_point(self, breast_cancer: BreastCancer) -> None:
        X, y = breast_cancer.full
        m = xgb.DMatrix(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            check_point = xgb.callback.TrainingCheckPoint(
                directory=tmpdir, interval=1, name="model"
            )
            xgb.train(
                {"objective": "binary:logistic"},
                m,
                num_boost_round=10,
                verbose_eval=False,
                callbacks=[check_point],
            )
            for i in range(1, 10):
                assert os.path.exists(
                    os.path.join(
                        tmpdir,
                        f"model_{i}.{xgb.callback.TrainingCheckPoint.default_format}",
                    )
                )

            check_point = xgb.callback.TrainingCheckPoint(
                directory=tmpdir, interval=1, as_pickle=True, name="model"
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

    def test_callback_list(self) -> None:
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

    def test_attribute_error(self, breast_cancer: BreastCancer) -> None:
        X, y = breast_cancer.full

        clf = xgb.XGBClassifier(n_estimators=8)
        clf.fit(X, y, eval_set=[(X, y)])

        with pytest.raises(AttributeError, match="early stopping is used"):
            clf.best_iteration

        with pytest.raises(AttributeError, match="early stopping is used"):
            clf.best_score

        booster = clf.get_booster()
        with pytest.raises(AttributeError, match="early stopping is used"):
            booster.best_iteration

        with pytest.raises(AttributeError, match="early stopping is used"):
            booster.best_score
