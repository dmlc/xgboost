"""Tests for running inplace prediction."""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Type, Union

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.data import get_california_housing, np_dtypes, pd_dtypes
from xgboost.testing.predict import run_base_margin_vs_base_score, run_predict_leaf


def test_inplace_predict_masked_array() -> None:
    data = np.array([[1.0, 2.0], [3.0, 999.0], [5.0, 6.0]], dtype=np.float32)
    dtrain = xgb.DMatrix(data, label=[0.0, 1.0, 0.0])
    booster = xgb.train({"tree_method": "hist", "max_depth": 1}, dtrain, 1)
    masked = np.ma.array(data, mask=[[False, False], [False, True], [False, False]])
    with pytest.raises(ValueError, match="Masked array is not supported"):
        booster.inplace_predict(masked)

    np.testing.assert_array_equal(
        booster.inplace_predict(np.ma.array(data)), booster.inplace_predict(data)
    )


def run_threaded_predict(X, rows, predict_func):
    results = []
    per_thread = 20
    with ThreadPoolExecutor(max_workers=10) as e:
        for i in range(0, rows, int(rows / per_thread)):
            if hasattr(X, "iloc"):
                predictor = X.iloc[i : i + per_thread, :]
            else:
                predictor = X[i : i + per_thread, ...]
            f = e.submit(predict_func, predictor)
            results.append(f)

    for f in results:
        assert f.result()


@pytest.mark.parametrize(
    "DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix, xgb.ExtMemQuantileDMatrix]
)
def test_predict_leaf(DMatrixT: Type[xgb.DMatrix]) -> None:
    run_predict_leaf("cpu", DMatrixT)


def test_predict_shape():
    X, y = get_california_housing()
    reg = xgb.XGBRegressor(n_estimators=1)
    reg.fit(X, y)
    predt = reg.get_booster().predict(xgb.DMatrix(X), strict_shape=True)
    assert len(predt.shape) == 2
    assert predt.shape[0] == X.shape[0]
    assert predt.shape[1] == 1

    contrib = reg.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True, strict_shape=True
    )
    assert len(contrib.shape) == 3
    assert contrib.shape[1] == 1

    contrib = reg.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True, approx_contribs=True
    )
    assert len(contrib.shape) == 2
    assert contrib.shape[1] == X.shape[1] + 1

    interaction = reg.get_booster().predict(
        xgb.DMatrix(X), pred_interactions=True, approx_contribs=True
    )
    assert len(interaction.shape) == 3
    assert interaction.shape[1] == X.shape[1] + 1
    assert interaction.shape[2] == X.shape[1] + 1

    interaction = reg.get_booster().predict(
        xgb.DMatrix(X), pred_interactions=True, approx_contribs=True, strict_shape=True
    )
    assert len(interaction.shape) == 4
    assert interaction.shape[1] == 1
    assert interaction.shape[2] == X.shape[1] + 1
    assert interaction.shape[3] == X.shape[1] + 1


def test_base_margin_vs_base_score() -> None:
    run_base_margin_vs_base_score("cpu")


@pytest.mark.parametrize("strict_shape", [False, True])
@pytest.mark.parametrize("booster", ["gbtree", "dart", "gblinear"])
@pytest.mark.parametrize(
    "objective",
    [
        "reg:squarederror",
        "binary:logistic",
        "multi:softprob",
        "multi:softmax",
        "reg:quantileerror",
    ],
)
def test_predict_empty_shape(strict_shape: bool, booster: str, objective: str) -> None:
    """Empty predictions preserve the trailing dimensions of non-empty predictions."""
    rng = np.random.default_rng(2026)
    X = rng.random((24, 4))
    y = rng.integers(0, 3 if objective.startswith("multi:") else 2, X.shape[0])
    params = {"objective": objective, "booster": booster, "nthread": 2}
    if objective.startswith("multi:"):
        params["num_class"] = 3
    if objective == "reg:quantileerror":
        params["quantile_alpha"] = [0.25, 0.75]
    if booster == "gblinear" and objective == "reg:quantileerror":
        pytest.skip("Quantile regression requires a tree booster")
    model = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=3)
    empty = X[:0]
    d_empty, d_full = xgb.DMatrix(empty), xgb.DMatrix(X[:4])
    width = (
        3
        if objective == "multi:softprob"
        else 2 if objective == "reg:quantileerror" else 1
    )
    for margin in [False, True]:
        expected_width = 3 if margin and objective.startswith("multi:") else width
        expected = (0, expected_width) if strict_shape or expected_width > 1 else (0,)
        kwargs = {"output_margin": margin, "strict_shape": strict_shape}
        result = model.predict(d_empty, **kwargs)
        assert result.shape == expected
        assert result.shape == (0,) + model.predict(d_full, **kwargs).shape[1:]
        if booster != "gblinear":
            inplace = {
                "predict_type": "margin" if margin else "value",
                "strict_shape": strict_shape,
            }
            assert model.inplace_predict(empty, **inplace).shape == expected
            assert expected == (0,) + model.inplace_predict(X[:4], **inplace).shape[1:]

    for interaction in [False, True]:
        for approximate in [False, True]:
            kwargs = {
                "pred_interactions" if interaction else "pred_contribs": True,
                "approx_contribs": approximate,
                "strict_shape": strict_shape,
            }
            result = model.predict(d_empty, **kwargs)
            assert result.shape == (0,) + model.predict(d_full, **kwargs).shape[1:]
            assert result.shape[-1] == X.shape[1] + 1


@pytest.mark.parametrize("strict_shape", [False, True])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("parallel_trees", [1, 3])
@pytest.mark.parametrize("rounds", [0, 1, 3])
def test_predict_empty_leaf_shape(
    strict_shape: bool, groups: int, parallel_trees: int, rounds: int
) -> None:
    """Leaf shape comes from the selected trees, including an empty forest."""
    rng = np.random.default_rng(2026)
    X = rng.random((24, 4))
    y = rng.integers(0, groups, X.shape[0])
    params = {"nthread": 2, "num_parallel_tree": parallel_trees}
    if groups > 1:
        params.update(objective="multi:softprob", num_class=groups)
    model = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=rounds)
    for end in sorted({0, min(rounds, 1)}):
        selected = end or rounds
        kwargs = {
            "pred_leaf": True,
            "strict_shape": strict_shape,
            "iteration_range": (0, end),
        }
        result = model.predict(xgb.DMatrix(X[:0]), **kwargs)
        assert (
            result.shape == (0,) + model.predict(xgb.DMatrix(X[:4]), **kwargs).shape[1:]
        )
        trees = selected * groups * parallel_trees
        if strict_shape:
            assert result.shape == (0, selected, groups, parallel_trees)
        else:
            assert result.shape == ((0,) if trees == 1 else (0, trees))


@pytest.mark.parametrize("strict_shape", [False, True])
@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_predict_empty_multi_target_shape(strict_shape: bool, strategy: str) -> None:
    rng = np.random.default_rng(2026)
    X, y = rng.random((24, 4)), rng.random((24, 2))
    model = xgb.train(
        {"nthread": 2, "tree_method": "hist", "multi_strategy": strategy},
        xgb.DMatrix(X, label=y),
        num_boost_round=2,
    )
    for margin in [False, True]:
        assert model.predict(
            xgb.DMatrix(X[:0]), output_margin=margin, strict_shape=strict_shape
        ).shape == (0, 2)
        assert model.inplace_predict(
            X[:0],
            predict_type="margin" if margin else "value",
            strict_shape=strict_shape,
        ).shape == (0, 2)
    if strategy == "multi_output_tree":
        assert model.predict(xgb.DMatrix(X[:0]), pred_leaf=True).shape == (0, 2)
        with pytest.raises(xgb.core.XGBoostError, match="vector leaf"):
            model.predict(xgb.DMatrix(X[:0]), pred_leaf=True, strict_shape=True)


class TestInplacePredict:
    """Tests for running inplace prediction"""

    @classmethod
    def setup_class(cls):
        cls.rows = 1000
        cls.cols = 10

        cls.missing = 11  # set to integer for testing

        cls.rng = np.random.RandomState(1994)

        cls.X = cls.rng.randn(cls.rows, cls.cols)
        missing_idx = [i for i in range(0, cls.cols, 4)]
        cls.X[:, missing_idx] = cls.missing  # set to be missing

        cls.y = cls.rng.randn(cls.rows)

        dtrain = xgb.DMatrix(cls.X, cls.y)
        cls.test = xgb.DMatrix(cls.X[:10, ...], missing=cls.missing)

        cls.num_boost_round = 10
        cls.booster = xgb.train({"tree_method": "hist"}, dtrain, num_boost_round=10)

    def test_predict(self):
        booster = self.booster
        X = self.X
        test = self.test

        predt_from_array = booster.inplace_predict(X[:10, ...], missing=self.missing)
        predt_from_dmatrix = booster.predict(test)

        X_obj = X.copy().astype(object)

        assert X_obj.dtype.hasobject is True
        assert X.dtype.hasobject is False
        np.testing.assert_allclose(
            booster.inplace_predict(X_obj), booster.inplace_predict(X)
        )

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        predt_from_array = booster.inplace_predict(
            X[:10, ...], iteration_range=(0, 4), missing=self.missing
        )
        predt_from_dmatrix = booster.predict(test, iteration_range=(0, 4))

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        with pytest.raises(ValueError):
            booster.predict(test, iteration_range=(0, booster.num_boosted_rounds() + 2))

        default = booster.predict(test)

        range_full = booster.predict(test, iteration_range=(0, self.num_boost_round))
        np.testing.assert_allclose(range_full, default)

        range_full = booster.predict(
            test, iteration_range=(0, booster.num_boosted_rounds())
        )
        np.testing.assert_allclose(range_full, default)

        def predict_dense(x):
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, self.rows, predict_dense)

        def predict_csr(x):
            inplace_predt = booster.inplace_predict(sparse.csr_matrix(x))
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, self.rows, predict_csr)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_predict_pd(self):
        X = self.X
        # construct it in column major style
        df = pd.DataFrame({str(i): X[:, i] for i in range(X.shape[1])})
        booster = self.booster
        df_predt = booster.inplace_predict(df)
        arr_predt = booster.inplace_predict(X)
        dmat_predt = booster.predict(xgb.DMatrix(X))

        X = df.values
        X = np.asfortranarray(X)
        fort_predt = booster.inplace_predict(X)

        np.testing.assert_allclose(dmat_predt, arr_predt)
        np.testing.assert_allclose(df_predt, arr_predt)
        np.testing.assert_allclose(fort_predt, arr_predt)

    def test_base_margin(self):
        booster = self.booster

        base_margin = self.rng.randn(self.rows)
        from_inplace = booster.inplace_predict(data=self.X, base_margin=base_margin)

        dtrain = xgb.DMatrix(self.X, self.y, base_margin=base_margin)
        from_dmatrix = booster.predict(dtrain)
        np.testing.assert_allclose(from_dmatrix, from_inplace)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_dtypes(self) -> None:
        for orig, x in np_dtypes(self.rows, self.cols):
            predt_orig = self.booster.inplace_predict(orig)
            predt = self.booster.inplace_predict(x)
            np.testing.assert_allclose(predt, predt_orig)

        # unsupported types
        for dtype in [
            np.bytes_,
            np.complex64,
            np.complex128,
        ]:
            X: np.ndarray = np.array(orig, dtype=dtype)
            with pytest.raises(ValueError):
                self.booster.inplace_predict(X)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_pd_dtypes(self) -> None:
        import pandas as pd
        from pandas.api.types import is_bool_dtype

        for orig, x in pd_dtypes():
            dtypes: Union[List, pd.Series] = (
                orig.dtypes if isinstance(orig, pd.DataFrame) else [orig.dtypes]
            )
            if isinstance(orig, pd.DataFrame) and is_bool_dtype(dtypes.iloc[0]):
                continue
            y = np.arange(x.shape[0])
            Xy = xgb.DMatrix(orig, y)
            booster = xgb.train({"tree_method": "hist"}, Xy, num_boost_round=1)
            predt_orig = booster.inplace_predict(orig)
            predt = booster.inplace_predict(x)
            np.testing.assert_allclose(predt, predt_orig)
