import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.metrics import (
    check_precision_score,
    check_quantile_error,
    run_pr_auc_binary,
    run_pr_auc_ltr,
    run_pr_auc_multi,
    run_roc_auc_binary,
    run_roc_auc_multi,
)

rng = np.random.RandomState(1337)


class TestEvalMetrics:
    xgb_params_01 = {"nthread": 1, "eval_metric": "error"}

    xgb_params_02 = {"nthread": 1, "eval_metric": ["error"]}

    xgb_params_03 = {"nthread": 1, "eval_metric": ["rmse", "error"]}

    xgb_params_04 = {"nthread": 1, "eval_metric": ["error", "rmse"]}

    def evalerror_01(self, preds, dtrain):
        labels = dtrain.get_label()
        return "error", float(sum(labels != (preds > 0.0))) / len(labels)

    def evalerror_02(self, preds, dtrain):
        labels = dtrain.get_label()
        return [("error", float(sum(labels != (preds > 0.0))) / len(labels))]

    @pytest.mark.skipif(**tm.no_sklearn())
    def evalerror_03(self, preds, dtrain):
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        return [
            ("rmse", mean_squared_error(labels, preds)),
            ("error", float(sum(labels != (preds > 0.0))) / len(labels)),
        ]

    @pytest.mark.skipif(**tm.no_sklearn())
    def evalerror_04(self, preds, dtrain):
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        return [
            ("error", float(sum(labels != (preds > 0.0))) / len(labels)),
            ("rmse", mean_squared_error(labels, preds)),
        ]

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_eval_metrics(self):
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            from sklearn.cross_validation import train_test_split
        from sklearn.datasets import load_digits

        digits = load_digits(n_class=2)
        X = digits["data"]
        y = digits["target"]

        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=0)

        dtrain = xgb.DMatrix(Xt, label=yt)
        dvalid = xgb.DMatrix(Xv, label=yv)

        watchlist = [(dtrain, "train"), (dvalid, "val")]

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain, num_boost_round=10)
        gbdt_02 = xgb.train(self.xgb_params_02, dtrain, num_boost_round=10)
        gbdt_03 = xgb.train(self.xgb_params_03, dtrain, num_boost_round=10)
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]

        gbdt_01 = xgb.train(
            self.xgb_params_01, dtrain, 10, watchlist, early_stopping_rounds=2
        )
        gbdt_02 = xgb.train(
            self.xgb_params_02, dtrain, 10, watchlist, early_stopping_rounds=2
        )
        gbdt_03 = xgb.train(
            self.xgb_params_03, dtrain, 10, watchlist, early_stopping_rounds=2
        )
        gbdt_04 = xgb.train(
            self.xgb_params_04, dtrain, 10, watchlist, early_stopping_rounds=2
        )
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]
        assert gbdt_03.predict(dvalid)[0] != gbdt_04.predict(dvalid)[0]

        gbdt_01 = xgb.train(
            self.xgb_params_01,
            dtrain,
            10,
            watchlist,
            early_stopping_rounds=2,
            custom_metric=self.evalerror_01,
        )
        gbdt_02 = xgb.train(
            self.xgb_params_02,
            dtrain,
            10,
            watchlist,
            early_stopping_rounds=2,
            custom_metric=self.evalerror_02,
        )
        gbdt_03 = xgb.train(
            self.xgb_params_03,
            dtrain,
            10,
            watchlist,
            early_stopping_rounds=2,
            custom_metric=self.evalerror_03,
        )
        gbdt_04 = xgb.train(
            self.xgb_params_04,
            dtrain,
            10,
            watchlist,
            early_stopping_rounds=2,
            custom_metric=self.evalerror_04,
        )
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]
        assert gbdt_03.predict(dvalid)[0] != gbdt_04.predict(dvalid)[0]

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_gamma_deviance(self):
        from sklearn.metrics import mean_gamma_deviance

        rng = np.random.RandomState(1994)
        n_samples = 100
        n_features = 30

        X = rng.randn(n_samples, n_features)
        y = rng.randn(n_samples)
        y = y - y.min() * 100

        reg = xgb.XGBRegressor(
            tree_method="hist",
            objective="reg:gamma",
            n_estimators=10,
            eval_metric="gamma-deviance",
        )
        reg.fit(X, y)

        booster = reg.get_booster()
        score = reg.predict(X)
        gamma_dev = float(booster.eval(xgb.DMatrix(X, y)).split(":")[1].split(":")[0])
        skl_gamma_dev = mean_gamma_deviance(y, score)
        np.testing.assert_allclose(gamma_dev, skl_gamma_dev, atol=1e-6)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_gamma_lik(self) -> None:
        import scipy.stats as stats

        rng = np.random.default_rng(1994)
        n_samples = 32
        n_features = 10

        X = rng.normal(0, 1, size=n_samples * n_features).reshape(
            (n_samples, n_features)
        )

        alpha, loc, beta = 5.0, 11.1, 22
        y = stats.gamma.rvs(
            alpha, loc=loc, scale=beta, size=n_samples, random_state=rng
        )
        reg = xgb.XGBRegressor(
            tree_method="hist",
            objective="reg:gamma",
            n_estimators=64,
            eval_metric="gamma-nloglik",
        )
        reg.fit(X, y, eval_set=[(X, y)])

        score = reg.predict(X)

        booster = reg.get_booster()
        nloglik = float(booster.eval(xgb.DMatrix(X, y)).split(":")[1].split(":")[0])

        # \beta_i = - (1 / \theta_i a)
        # where \theta_i is the canonical parameter
        # XGBoost uses the canonical link function of gamma in evaluation function.
        # so \theta = - (1.0 / y)
        # dispersion is hardcoded as 1.0, so shape (a in scipy parameter) is also 1.0
        beta = -(1.0 / (-(1.0 / y)))  # == y
        nloglik_stats = -stats.gamma.logpdf(score, a=1.0, scale=beta)

        np.testing.assert_allclose(nloglik, np.mean(nloglik_stats), rtol=1e-3)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_roc_auc(self, n_samples: int) -> None:
        run_roc_auc_binary("hist", n_samples, "cpu")

    @pytest.mark.parametrize(
        "n_samples,weighted", [(4, False), (100, False), (1000, False), (10000, True)]
    )
    def test_roc_auc_multi(self, n_samples: int, weighted: bool) -> None:
        run_roc_auc_multi("hist", n_samples, weighted, "cpu")

    def test_pr_auc_binary(self) -> None:
        run_pr_auc_binary("hist", "cpu")

    def test_pr_auc_multi(self) -> None:
        run_pr_auc_multi("hist", "cpu")

    def test_pr_auc_ltr(self) -> None:
        run_pr_auc_ltr("hist", "cpu")

    def test_precision_score(self) -> None:
        check_precision_score("hist", "cpu")

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_quantile_error(self) -> None:
        check_quantile_error("hist", "cpu")
