import xgboost as xgb
import testing as tm
import numpy as np
import pytest

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
            num_boost_round=8,
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
    @pytest.mark.parametrize("n_samples", [4, 100, 1000])
    def test_roc_auc(self, n_samples):
        self.run_roc_auc_binary("hist", n_samples)

    def run_roc_auc_multi(self, tree_method, n_samples):
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

        Xy = xgb.DMatrix(X, y)
        booster = xgb.train(
            {
                "tree_method": tree_method,
                "eval_metric": "auc",
                "objective": "multi:softprob",
                "num_class": n_classes,
            },
            Xy,
            num_boost_round=8,
        )
        score = booster.predict(Xy)
        skl_auc = roc_auc_score(y, score, average="weighted", multi_class="ovr")
        auc = float(booster.eval(Xy).split(":")[1])
        np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)

        X = rng.randn(*X.shape)
        score = booster.predict(xgb.DMatrix(X))
        skl_auc = roc_auc_score(y, score, average="weighted", multi_class="ovr")
        auc = float(booster.eval(xgb.DMatrix(X, y)).split(":")[1])
        np.testing.assert_allclose(skl_auc, auc, rtol=1e-6)

    @pytest.mark.parametrize("n_samples", [4, 100, 1000])
    def test_roc_auc_multi(self, n_samples):
        self.run_roc_auc_multi("hist", n_samples)
