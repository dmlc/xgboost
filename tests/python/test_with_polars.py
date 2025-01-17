"""Copyright 2024, XGBoost contributors"""

import json
import os
import tempfile
from typing import Type, Union

import numpy as np
import pytest

import xgboost as xgb

pl = pytest.importorskip("polars")


@pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
def test_polars_basic(
    DMatrixT: Union[Type[xgb.DMatrix], Type[xgb.QuantileDMatrix]]
) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    Xy = DMatrixT(df)
    assert Xy.num_row() == df.shape[0]
    assert Xy.num_col() == df.shape[1]
    assert Xy.num_nonmissing() == np.prod(df.shape)

    # feature info
    assert Xy.feature_names == df.columns
    assert Xy.feature_types == ["int", "int"]

    res = Xy.get_data().toarray()
    res1 = df.to_numpy()

    if isinstance(Xy, xgb.QuantileDMatrix):
        # skip min values in the cut.
        np.testing.assert_allclose(res[1:, :], res1[1:, :])
    else:
        np.testing.assert_allclose(res, res1)

    # boolean
    df = pl.DataFrame({"a": [True, False, False], "b": [False, False, True]})
    Xy = DMatrixT(df)
    np.testing.assert_allclose(
        Xy.get_data().data, np.array([1, 0, 0, 0, 0, 1]), atol=1e-5
    )


def test_polars_missing() -> None:
    df = pl.DataFrame({"a": [1, None, 3], "b": [3, 4, None]})
    Xy = xgb.DMatrix(df)
    assert Xy.num_row() == df.shape[0]
    assert Xy.num_col() == df.shape[1]
    assert Xy.num_nonmissing() == 4

    np.testing.assert_allclose(Xy.get_data().data, np.array([1, 3, 4, 3]))
    np.testing.assert_allclose(Xy.get_data().indptr, np.array([0, 2, 3, 4]))
    np.testing.assert_allclose(Xy.get_data().indices, np.array([0, 1, 1, 0]))

    ser = pl.Series("y", np.arange(0, df.shape[0]))
    Xy.set_info(label=ser)
    booster = xgb.train({}, Xy, num_boost_round=1)
    predt0 = booster.inplace_predict(df)
    predt1 = booster.predict(Xy)
    np.testing.assert_allclose(predt0, predt1)


def test_classififer() -> None:
    from sklearn.datasets import make_classification, make_multilabel_classification

    X, y = make_classification(random_state=2024)
    X_df = pl.DataFrame(X)
    y_ser = pl.Series(y)

    clf0 = xgb.XGBClassifier()
    clf0.fit(X_df, y_ser)

    clf1 = xgb.XGBClassifier()
    clf1.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path0 = os.path.join(tmpdir, "clf0.json")
        clf0.save_model(path0)

        path1 = os.path.join(tmpdir, "clf1.json")
        clf1.save_model(path1)

        with open(path0, "r") as fd:
            model0 = json.load(fd)
        with open(path1, "r") as fd:
            model1 = json.load(fd)

    model0["learner"]["feature_names"] = []
    model0["learner"]["feature_types"] = []
    assert model0 == model1

    predt0 = clf0.predict(X)
    predt1 = clf1.predict(X)

    np.testing.assert_allclose(predt0, predt1)

    assert (clf0.feature_names_in_ == X_df.columns).all()
    assert clf0.n_features_in_ == X_df.shape[1]

    X, y = make_multilabel_classification(128)
    X_df = pl.DataFrame(X)
    y_df = pl.DataFrame(y)
    clf = xgb.XGBClassifier(n_estimators=1)
    clf.fit(X_df, y_df)
    assert clf.n_classes_ == 2

    X, y = make_classification(n_classes=3, n_informative=5)
    X_df = pl.DataFrame(X)
    y_ser = pl.Series(y)
    clf = xgb.XGBClassifier(n_estimators=1)
    clf.fit(X_df, y_ser)
    assert clf.n_classes_ == 3


def test_regressor() -> None:
    from sklearn.datasets import make_regression

    X, y = make_regression(n_targets=3)
    X_df = pl.DataFrame(X)
    y_df = pl.DataFrame(y)
    assert y_df.shape[1] == 3

    reg0 = xgb.XGBRegressor()
    reg0.fit(X_df, y_df)

    reg1 = xgb.XGBRegressor()
    reg1.fit(X, y)

    predt0 = reg0.predict(X)
    predt1 = reg1.predict(X)

    np.testing.assert_allclose(predt0, predt1)

def test_categorical() ->  None:
    import polars as pl

    df = pl.DataFrame(
        {"f0": [1, 2, 3], "b": ["a", "b", "c"]},
        schema=[("a", pl.Int64()), ("b", pl.Categorical())]
    )
    with pytest.raises(NotImplementedError, match="Categorical feature"):
        xgb.DMatrix(df, enable_categorical=True)
