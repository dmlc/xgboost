import numpy as np
import pytest
import xgboost as xgb
from xgboost import interpret


def test_shap_values_matches_predict() -> None:
    rng = np.random.RandomState(1994)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    booster = xgb.train({"tree_method": "hist"}, xgb.DMatrix(X, label=y), 4)

    values, bias = interpret.shap_values(booster, X, return_bias=True)
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)

    np.testing.assert_allclose(values, contribs[:, :-1])
    np.testing.assert_allclose(bias, contribs[:, -1])
    np.testing.assert_allclose(interpret.shap_values(booster, X), contribs[:, :-1])


def test_shap_values_accepts_sklearn_model() -> None:
    rng = np.random.RandomState(1995)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    reg = xgb.XGBRegressor(n_estimators=4, tree_method="hist")
    reg.fit(X, y)

    values = interpret.shap_values(reg, X)
    contribs = reg.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)

    np.testing.assert_allclose(values, contribs[:, :-1])


def test_shap_values_rejects_background_data() -> None:
    rng = np.random.RandomState(1996)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    booster = xgb.train({"tree_method": "hist"}, xgb.DMatrix(X, label=y), 4)

    with pytest.raises(NotImplementedError, match="X_background"):
        interpret.shap_values(booster, X, X_background=X)
