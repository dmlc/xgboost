import numpy as np
import pytest
import xgboost as xgb
from xgboost import interpret


def test_shap_values_matches_predict() -> None:
    rng = np.random.RandomState(1994)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    booster = xgb.train({"tree_method": "hist"}, xgb.DMatrix(X, label=y), 4)

    values, bias = interpret.shap_values(booster, X)
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)

    np.testing.assert_allclose(values, contribs[:, :-1])
    np.testing.assert_allclose(bias, contribs[:, -1])


def test_shap_values_accepts_sklearn_model() -> None:
    rng = np.random.RandomState(1995)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    reg = xgb.XGBRegressor(n_estimators=4, tree_method="hist")
    reg.fit(X, y)

    values, bias = interpret.shap_values(reg, X)
    contribs = reg.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)

    np.testing.assert_allclose(values, contribs[:, :-1])
    np.testing.assert_allclose(bias, contribs[:, -1])


def test_shap_values_uses_sklearn_iteration_range() -> None:
    rng = np.random.RandomState(1996)
    X = rng.randn(64, 4)
    y = rng.randn(64)
    reg = xgb.XGBRegressor(n_estimators=8, tree_method="hist")
    reg.fit(X, y)
    reg.get_booster().set_attr(best_iteration="3")

    values, bias = interpret.shap_values(reg, X, iteration_range=(0, 0))
    contribs = reg.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, 4)
    )

    np.testing.assert_allclose(values, contribs[:, :-1])
    np.testing.assert_allclose(bias, contribs[:, -1])


def test_shap_values_rejects_background_data() -> None:
    rng = np.random.RandomState(1997)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    booster = xgb.train({"tree_method": "hist"}, xgb.DMatrix(X, label=y), 4)

    with pytest.raises(NotImplementedError, match="X_background"):
        interpret.shap_values(booster, X, X_background=X)


def test_shap_values_validates_get_booster() -> None:
    class InvalidModel:
        get_booster = "booster"

    with pytest.raises(TypeError, match="get_booster"):
        interpret.shap_values(InvalidModel(), np.empty((1, 1)))


def test_shap_values_uses_missing_for_array_like_data() -> None:
    X = np.array([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0, 1.0])
    booster = xgb.train(
        {"tree_method": "hist"}, xgb.DMatrix(X, label=y, missing=0.0), 4
    )

    values, bias = interpret.shap_values(booster, X, missing=0.0)
    contribs = booster.predict(xgb.DMatrix(X, missing=0.0), pred_contribs=True)

    np.testing.assert_allclose(values, contribs[:, :-1])
    np.testing.assert_allclose(bias, contribs[:, -1])


def test_shap_values_rejects_missing_with_dmatrix() -> None:
    X = xgb.DMatrix(np.array([[0.0, 1.0]]), label=np.array([0.0]), missing=0.0)
    booster = xgb.train({"tree_method": "hist"}, X, 1)

    with pytest.raises(ValueError, match="DMatrix"):
        interpret.shap_values(booster, X, missing=0.0)


def test_shap_values_device_override_restores_config() -> None:
    rng = np.random.RandomState(1998)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    booster = xgb.train({"tree_method": "hist"}, xgb.DMatrix(X, label=y), 4)
    config = booster.save_config()

    values, bias = interpret.shap_values(booster, X, device="cpu")
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)

    np.testing.assert_allclose(values, contribs[:, :-1])
    np.testing.assert_allclose(bias, contribs[:, -1])
    assert booster.save_config() == config


def test_shap_values_device_override_restores_config_on_error() -> None:
    rng = np.random.RandomState(1999)
    X = rng.randn(16, 4)
    y = rng.randn(16)
    booster = xgb.train(
        {"tree_method": "hist"},
        xgb.DMatrix(X, label=y, feature_names=["a", "b", "c", "d"]),
        4,
    )
    config = booster.save_config()

    with pytest.raises(ValueError, match="feature_names mismatch"):
        interpret.shap_values(
            booster,
            xgb.DMatrix(X, feature_names=["q", "b", "c", "d"]),
            device="cpu",
        )

    assert booster.save_config() == config
