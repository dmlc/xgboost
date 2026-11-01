"""Copyright 2026, XGBoost contributors"""

import re
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import pytest

import xgboost as xgb


@pytest.fixture(params=["arrow", "polars", "pandas"])
def frame(request: pytest.FixtureRequest) -> Callable[[np.ndarray, list[str]], Any]:
    if request.param == "arrow":
        pa = pytest.importorskip("pyarrow")
        return lambda values, names: pa.table(
            {name: values[:, i] for i, name in enumerate(names)}
        )
    if request.param == "polars":
        pytest.importorskip("pyarrow")
        pl = pytest.importorskip("polars")
        return lambda values, names: pl.DataFrame(
            {name: values[:, i] for i, name in enumerate(names)}
        )
    pd = pytest.importorskip("pandas")
    return lambda values, names: pd.DataFrame(values, columns=names)


@pytest.fixture
def values() -> np.ndarray:
    return np.array(
        [[0, 3], [1, 2], [2, 1], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]],
        dtype=np.float32,
    )


def train(values: np.ndarray, names: list[str] | None) -> xgb.Booster:
    dtrain = xgb.DMatrix(values, label=values[:, 0], feature_names=names, nthread=1)
    return xgb.train(
        {
            "booster": "gbtree",
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": "cpu",
            "nthread": 1,
            "max_depth": 2,
            "seed": 0,
        },
        dtrain,
        num_boost_round=3,
    )


@pytest.fixture
def booster(values: np.ndarray) -> xgb.Booster:
    return train(values, ["a", "b"])


def test_matching_names(
    frame: Callable[[np.ndarray, list[str]], Any],
    values: np.ndarray,
    booster: xgb.Booster,
) -> None:
    data = frame(values, ["a", "b"])
    expected = booster.predict(xgb.DMatrix(data, nthread=1))
    np.testing.assert_allclose(booster.inplace_predict(data), expected)
    np.testing.assert_allclose(
        booster.inplace_predict(data, validate_features=True), expected
    )


@pytest.mark.parametrize("change", ["reordered", "renamed"])
def test_mismatched_names(
    frame: Callable[[np.ndarray, list[str]], Any],
    values: np.ndarray,
    booster: xgb.Booster,
    change: str,
) -> None:
    if change == "reordered":
        values = values[:, ::-1].copy()
        names = ["b", "a"]
    else:
        names = ["a", "other"]
    data = frame(values, names)

    # Disabling validation retains positional prediction for named containers.
    expected = booster.inplace_predict(values, validate_features=False)
    np.testing.assert_allclose(
        booster.inplace_predict(data, validate_features=False), expected
    )
    # The regular prediction API already enforces the same contract.
    with pytest.raises(ValueError, match="feature_names mismatch"):
        booster.predict(xgb.DMatrix(data, nthread=1))
    with pytest.raises(ValueError, match="feature_names mismatch"):
        booster.inplace_predict(data)
    with pytest.raises(ValueError, match="feature_names mismatch"):
        booster.inplace_predict(data, validate_features=True)


def test_unnamed_booster(
    frame: Callable[[np.ndarray, list[str]], Any], values: np.ndarray
) -> None:
    booster = train(values, None)
    assert booster.feature_names is None
    data = frame(values, ["a", "b"])
    np.testing.assert_allclose(
        booster.inplace_predict(data), booster.inplace_predict(values)
    )
    # Inplace prediction must not acquire feature names as a side effect.
    assert booster.feature_names is None
    with pytest.raises(ValueError, match="Feature shape mismatch"):
        booster.inplace_predict(frame(values[:, :1].copy(), ["a"]))


def test_numpy_validation(values: np.ndarray, booster: xgb.Booster) -> None:
    # NumPy has no column names: the existing inplace API checks only its shape.
    expected = booster.predict(xgb.DMatrix(values, feature_names=["a", "b"], nthread=1))
    np.testing.assert_allclose(booster.inplace_predict(values), expected)
    with pytest.raises(ValueError, match="Feature shape mismatch"):
        booster.inplace_predict(values[:, :1].copy())


@pytest.mark.parametrize("container", ["series", "lazyframe"])
def test_polars_single_column_names(values: np.ndarray, container: str) -> None:
    pytest.importorskip("pyarrow")
    pl = pytest.importorskip("polars")
    values = values[:, :1].copy()
    booster = train(values, ["a"])

    def make_data(name: str) -> Any:
        if container == "series":
            return pl.Series(name, values[:, 0])
        return pl.DataFrame({name: values[:, 0]}).lazy()

    def predict(data: Any, **kwargs: Any) -> np.ndarray:
        warning = (
            "Using the default parameters for the polars `LazyFrame.collect`. Consider"
            " passing a realized `DataFrame` or `Series` instead."
        )
        context = (
            pytest.warns(UserWarning, match="^" + re.escape(warning) + "$")
            if container == "lazyframe"
            else nullcontext()
        )
        with context:
            return booster.inplace_predict(data, **kwargs)

    expected = booster.inplace_predict(values)
    np.testing.assert_allclose(predict(make_data("a")), expected)
    renamed = make_data("other")
    np.testing.assert_allclose(predict(renamed, validate_features=False), expected)
    with pytest.raises(ValueError, match="feature_names mismatch"):
        predict(renamed)
    with pytest.raises(ValueError, match="feature_names mismatch"):
        predict(renamed, validate_features=True)
