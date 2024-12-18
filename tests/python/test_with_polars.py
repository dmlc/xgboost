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

    res = Xy.get_data().toarray()
    res1 = df.to_numpy()

    if isinstance(Xy, xgb.QuantileDMatrix):
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
