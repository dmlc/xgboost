import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.metrics import check_quantile_error

rng = np.random.RandomState(1337)


class TestEvalMetrics:
    xgb_params_01 = {
        'verbosity': 0,
        'nthread': 1,
        'eval_metric': 'error'
    }

    xgb_params_02 = {
        'verbosity': 0,
        'nthread': 1,
        'eval_metric': ['error']
    }

    xgb_params_03 = {
        'verbosity': 0,
        'nthread': 1,
        'eval_metric': ['rmse', 'error']
    }

    xgb_params_04 = {
        'verbosity': 0,
        'nthread': 1,
        'eval_metric': ['error', 'rmse']
    }

    def evalerror_01(self, preds, dtrain):
        labels = dtrain.get_label()
        return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

    def evalerror_02(self, preds, dtrain):
        labels = dtrain.get_label()
        return [('error', float(sum(labels != (preds > 0.0))) / len(labels))]

    @pytest.mark.skipif(**tm.no_sklearn())
    def evalerror_03(self, preds, dtrain):
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        return [('rmse', mean_squared_error(labels, preds)),
                ('error', float(sum(labels != (preds > 0.0))) / len(labels))]

    @pytest.mark.skipif(**tm.no_sklearn())
    def evalerror_04(self, preds, dtrain):
        from sklearn.metrics import mean_squared_error

        labels = dtrain.get_label()
        return [('error', float(sum(labels != (preds > 0.0))) / len(labels)),
                ('rmse', mean_squared_error(labels, preds))]

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_eval_metrics(self):
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            from sklearn.cross_validation import train_test_split
        from sklearn.datasets import load_digits

        digits = load_digits(n_class=2)
        X = digits['data']
        y = digits['target']

        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=0)

        dtrain = xgb.DMatrix(Xt, label=yt)
        dvalid = xgb.DMatrix(Xv, label=yv)

        watchlist = [(dtrain, 'train'), (dvalid, 'val')]

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain, num_boost_round=10)
        gbdt_02 = xgb.train(self.xgb_params_02, dtrain, num_boost_round=10)
        gbdt_03 = xgb.train(self.xgb_params_03, dtrain, num_boost_round=10)
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        gbdt_02 = xgb.train(self.xgb_params_02, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        gbdt_03 = xgb.train(self.xgb_params_03, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        gbdt_04 = xgb.train(self.xgb_params_04, dtrain, 10, watchlist,
                            early_stopping_rounds=2)
        assert gbdt_01.predict(dvalid)[0] == gbdt_02.predict(dvalid)[0]
        assert gbdt_01.predict(dvalid)[0] == gbdt_03.predict(dvalid)[0]
        assert gbdt_03.predict(dvalid)[0] != gbdt_04.predict(dvalid)[0]

        gbdt_01 = xgb.train(self.xgb_params_01, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_01)
        gbdt_02 = xgb.train(self.xgb_params_02, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_02)
        gbdt_03 = xgb.train(self.xgb_params_03, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_03)
        gbdt_04 = xgb.train(self.xgb_params_04, dtrain, 10, watchlist,
                            early_stopping_rounds=2, feval=self.evalerror_04)
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

        reg = xgb.XGBRegressor(tree_method="hist", objective="reg:gamma", n_estimators=10)
        reg.fit(X, y, eval_metric="gamma-deviance")

        booster = reg.get_booster()
        score = reg.predict(X)
        gamma_dev = float(booster.eval(xgb.DMatrix(X, y)).split(":")[1].split(":")[0])
        skl_gamma_dev = mean_gamma_deviance(y, score)
        np.testing.assert_allclose(gamma_dev, skl_gamma_dev, rtol=1e-6)

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_gamma_lik(self) -> None:
        import scipy.stats as stats
        rng = np.random.default_rng(1994)
        n_samples = 32
        n_features = 10

        X = rng.normal(0, 1, size=n_samples * n_features).reshape((n_samples, n_features))

        alpha, loc, beta = 5.0, 11.1, 22
        y = stats.gamma.rvs(alpha, loc=loc, scale=beta, size=n_samples, random_state=rng)
        reg = xgb.XGBRegressor(tree_method="hist", objective="reg:gamma", n_estimators=64)
        reg.fit(X, y, eval_metric="gamma-nloglik", eval_set=[(X, y)])

        score = reg.predict(X)

        booster = reg.get_booster()
        nloglik = float(booster.eval(xgb.DMatrix(X, y)).split(":")[1].split(":")[0])

        # \beta_i = - (1 / \theta_i a)
        # where \theta_i is the canonical parameter
        # XGBoost uses the canonical link function of gamma in evaluation function.
        # so \theta = - (1.0 / y)
        # dispersion is hardcoded as 1.0, so shape (a in scipy parameter) is also 1.0
        beta = - (1.0 / (- (1.0 / y)))  # == y
        nloglik_stats = -stats.gamma.logpdf(score, a=1.0, scale=beta)

        np.testing.assert_allclose(nloglik, np.mean(nloglik_stats), rtol=1e-3)

    def run_roc_auc_binary(self, tree_method, n_samples):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(1994)
        n_samples = n_samples
        n_features = 10

        X, y = make_classification(
            n_samples,
            n_features,
            n_informative=n_features,
            n_redundant=0,
            random_state=rng
        )
        Xy = xgb.DMatrix(X, y)
        booster = xgb.train(
            {
                "tree_method": tree_method,
                "eval_metric": "auc",
                "objective": "binary:logistic",
            },
            Xy,
            num_boost_round=1,
        )
        score = booster.predict(Xy)
        skl_auc = roc_auc_score(y, score)
        auc = float(booster.eval(Xy).split(":")[1])
        np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)

        X = rng.randn(*X.shape)
        score = booster.predict(xgb.DMatrix(X))
        skl_auc = roc_auc_score(y, score)
        auc = float(booster.eval(xgb.DMatrix(X, y)).split(":")[1])
        np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_roc_auc(self, n_samples):
        self.run_roc_auc_binary("hist", n_samples)

    def run_roc_auc_multi(self, tree_method, n_samples, weighted):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(1994)
        n_samples = n_samples
        n_features = 10
        n_classes = 4

        X, y = make_classification(
            n_samples,
            n_features,
            n_informative=n_features,
            n_redundant=0,
            n_classes=n_classes,
            random_state=rng
        )
        if weighted:
            weights = rng.randn(n_samples)
            weights -= weights.min()
            weights /= weights.max()
        else:
            weights = None

        Xy = xgb.DMatrix(X, y, weight=weights)
        booster = xgb.train(
            {
                "tree_method": tree_method,
                "eval_metric": "auc",
                "objective": "multi:softprob",
                "num_class": n_classes,
            },
            Xy,
            num_boost_round=1,
        )
        score = booster.predict(Xy)
        skl_auc = roc_auc_score(
            y, score, average="weighted", sample_weight=weights, multi_class="ovr"
        )
        auc = float(booster.eval(Xy).split(":")[1])
        np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)

        X = rng.randn(*X.shape)

        score = booster.predict(xgb.DMatrix(X, weight=weights))
        skl_auc = roc_auc_score(
            y, score, average="weighted", sample_weight=weights, multi_class="ovr"
        )
        auc = float(booster.eval(xgb.DMatrix(X, y, weight=weights)).split(":")[1])
        np.testing.assert_allclose(skl_auc, auc, rtol=1e-5)

    @pytest.mark.parametrize(
        "n_samples,weighted", [(4, False), (100, False), (1000, False), (10000, True)]
    )
    def test_roc_auc_multi(self, n_samples, weighted):
        self.run_roc_auc_multi("hist", n_samples, weighted)

    def run_pr_auc_binary(self, tree_method):
        from sklearn.datasets import make_classification
        from sklearn.metrics import auc, precision_recall_curve
        X, y = make_classification(128, 4, n_classes=2, random_state=1994)
        clf = xgb.XGBClassifier(tree_method=tree_method, n_estimators=1)
        clf.fit(X, y, eval_metric="aucpr", eval_set=[(X, y)])
        evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]

        y_score = clf.predict_proba(X)[:, 1]  # get the positive column
        precision, recall, _ = precision_recall_curve(y, y_score)
        prauc = auc(recall, precision)
        # Interpolation results are slightly different from sklearn, but overall should be
        # similar.
        np.testing.assert_allclose(prauc, evals_result, rtol=1e-2)

        clf = xgb.XGBClassifier(tree_method=tree_method, n_estimators=10)
        clf.fit(X, y, eval_metric="aucpr", eval_set=[(X, y)])
        evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]
        np.testing.assert_allclose(0.99, evals_result, rtol=1e-2)

    def test_pr_auc_binary(self):
        self.run_pr_auc_binary("hist")

    def run_pr_auc_multi(self, tree_method):
        from sklearn.datasets import make_classification
        X, y = make_classification(
            64, 16, n_informative=8, n_classes=3, random_state=1994
        )
        clf = xgb.XGBClassifier(tree_method=tree_method, n_estimators=1)
        clf.fit(X, y, eval_metric="aucpr", eval_set=[(X, y)])
        evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]
        # No available implementation for comparison, just check that XGBoost converges to
        # 1.0
        clf = xgb.XGBClassifier(tree_method=tree_method, n_estimators=10)
        clf.fit(X, y, eval_metric="aucpr", eval_set=[(X, y)])
        evals_result = clf.evals_result()["validation_0"]["aucpr"][-1]
        np.testing.assert_allclose(1.0, evals_result, rtol=1e-2)

    def test_pr_auc_multi(self):
        self.run_pr_auc_multi("hist")

    def run_pr_auc_ltr(self, tree_method):
        from sklearn.datasets import make_classification
        X, y = make_classification(128, 4, n_classes=2, random_state=1994)
        ltr = xgb.XGBRanker(tree_method=tree_method, n_estimators=16)
        groups = np.array([32, 32, 64])
        ltr.fit(
            X,
            y,
            group=groups,
            eval_set=[(X, y)],
            eval_group=[groups],
            eval_metric="aucpr",
        )
        results = ltr.evals_result()["validation_0"]["aucpr"]
        assert results[-1] >= 0.99

    def test_pr_auc_ltr(self):
        self.run_pr_auc_ltr("hist")

    @pytest.mark.skipif(**tm.no_sklearn())
    def test_quantile_error(self) -> None:
        check_quantile_error("hist")
