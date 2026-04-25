"""Copyright 2024, XGBoost contributors"""

import json
import os
import time
from pathlib import Path
from typing import Type, Union

import numpy as np
import pytest
import xgboost as xgb
from xgboost.compat import is_dataframe

pl = pytest.importorskip("polars")


def test_type_check() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    assert is_dataframe(df)
    assert is_dataframe(df["a"])


@pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
def test_polars_basic(
    DMatrixT: Union[Type[xgb.DMatrix], Type[xgb.QuantileDMatrix]],
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


def test_classififer(tmp_path: Path) -> None:
    from sklearn.datasets import make_classification, make_multilabel_classification

    X, y = make_classification(random_state=2024)
    X_df = pl.DataFrame(X)
    y_ser = pl.Series(y)

    clf0 = xgb.XGBClassifier()
    clf0.fit(X_df, y_ser)

    clf1 = xgb.XGBClassifier()
    clf1.fit(X, y)

    path0 = tmp_path / "clf0.json"
    clf0.save_model(path0)

    path1 = tmp_path / "clf1.json"
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


def test_categorical() -> None:
    import polars as pl

    cats = ["aa", "cc", "bb", "ee", "ee"]
    df = pl.DataFrame(
        {"f0": [1, 3, 2, 4, 4], "f1": cats},
        schema=[("f0", pl.Int64()), ("f1", pl.Categorical(ordering="lexical"))],
    )

    data = xgb.DMatrix(df)
    categories = data.get_categories(export_to_arrow=True)
    assert dict(categories.to_arrow())["f0"] is None
    f1 = dict(categories.to_arrow())["f1"]
    assert f1 is not None
    assert f1.to_pylist() == cats[:4]

    df = pl.DataFrame(
        {"f0": [1, 3, 2, 4, 4], "f1": cats},
        schema=[("f0", pl.Int64()), ("f1", pl.Enum(cats[:4]))],
    )
    data = xgb.DMatrix(df)
    categories = data.get_categories(export_to_arrow=True)
    assert dict(categories.to_arrow())["f0"] is None
    f1 = dict(categories.to_arrow())["f1"]
    assert f1 is not None
    assert f1.to_pylist() == cats[:4]

    rng = np.random.default_rng(2025)
    y = rng.normal(size=(df.shape[0]))
    Xy = xgb.QuantileDMatrix(df, y)
    booster = xgb.train({}, Xy, num_boost_round=8)
    predt_0 = booster.inplace_predict(df)

    df_rev = pl.DataFrame(
        {"f0": [1, 3, 2, 4, 4], "f1": cats},
        schema=[("f0", pl.Int64()), ("f1", pl.Enum(cats[:4][::-1]))],
    )
    predt_1 = booster.inplace_predict(df_rev)
    assert (
        df["f1"].cat.get_categories().to_list()
        != df_rev["f1"].cat.get_categories().to_list()
    )
    np.testing.assert_allclose(predt_0, predt_1)


def test_categorical_sparse_codes() -> None:
    """Regression test for AddCategories over-allocation with sparse dictionary codes.

    A polars Categorical built against a pre-populated global StringCache holds a few
    unique strings at very large physical codes. Asserts prediction parity against an
    unprimed baseline so a silent dictionary-misalignment bug surfaces here rather than
    via the platform-specific heap-corruption symptom.
    """
    cats = [f"cat_{i}" for i in range(16)]
    n_rows = 2048
    rng = np.random.default_rng(2026)
    cat_choices = rng.choice(cats, size=n_rows)
    f0 = rng.normal(size=n_rows)
    y = rng.normal(size=n_rows)
    train_params = {"tree_method": "hist", "seed": 0}

    with pl.StringCache():
        primer = pl.Series(
            "primer", [f"primer_{i}" for i in range(200_000)], dtype=pl.Categorical
        )
        _unused = primer
        df_primed = pl.DataFrame(
            {"f0": f0, "f1": pl.Series("f1", cat_choices, dtype=pl.Categorical)}
        )
        Xy_primed = xgb.QuantileDMatrix(df_primed, y, enable_categorical=True)
        booster_primed = xgb.train(train_params, Xy_primed, num_boost_round=4)
        predt_primed = booster_primed.inplace_predict(df_primed)

    with pl.StringCache():
        df_baseline = pl.DataFrame(
            {"f0": f0, "f1": pl.Series("f1", cat_choices, dtype=pl.Categorical)}
        )
        Xy_baseline = xgb.QuantileDMatrix(df_baseline, y, enable_categorical=True)
        booster_baseline = xgb.train(train_params, Xy_baseline, num_boost_round=4)
        predt_baseline = booster_baseline.inplace_predict(df_baseline)

    assert predt_primed.shape == (n_rows,)
    assert np.isfinite(predt_primed).all()
    # primed (sparse codes ~200k+) and baseline (codes 0..15) carry the same logical
    # rows; predictions must match -- a deviation means dictionary alignment is off
    np.testing.assert_allclose(predt_primed, predt_baseline, rtol=1e-6, atol=1e-6)


def test_categorical_model_save_load_roundtrip(tmp_path: Path) -> None:
    """Save/load round-trip for a model trained on sparse polars Categorical codes.

    Regression safety net for the CatBitField / cut_values / MaxNumBinPerFeat cleanup: the
    serialized tree splits store categorical bit fields sized by observed physical codes, and
    this test guarantees that a model saved by the current code reloads and predicts identically.
    """
    rng = np.random.default_rng(2027)
    with pl.StringCache():
        # bind the primer Series to a local so its StringCache entries outlive it
        primer = pl.Series(
            "primer", [f"primer_{i}" for i in range(20_000)], dtype=pl.Categorical
        )
        _unused = primer
        cats = [f"cat_{i:02d}" for i in range(12)]
        col = pl.Series("f0", rng.choice(cats, size=1024), dtype=pl.Categorical)
        df = pl.DataFrame({"f0": col, "f1": rng.normal(size=col.len())})
        y = rng.integers(0, 2, size=col.len())

        dtrain = xgb.QuantileDMatrix(df, y, enable_categorical=True)
        booster = xgb.train(
            {"tree_method": "hist", "objective": "binary:logistic"},
            dtrain,
            num_boost_round=6,
        )
        pred_before = booster.inplace_predict(df)

        model_path = tmp_path / "booster.ubj"
        booster.save_model(str(model_path))

        booster2 = xgb.Booster()
        booster2.load_model(str(model_path))
        pred_after = booster2.inplace_predict(df)

    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-6, atol=1e-6)


def test_categorical_many_eval_sets_share_ref() -> None:
    """Prediction parity across multiple val DMatrices sharing one train as reference.

    Builds the booster once, then predicts on each val DMatrix built with ref=train
    and cross-checks against inplace_predict on the raw polars DataFrame. A silent
    alias failure (val DMatrix acquiring a fresh empty cats instead of the train's
    dictionary) would diverge here.
    """
    rng = np.random.default_rng(2028)
    with pl.StringCache():
        cats = [f"cat_{i:02d}" for i in range(8)]
        df_train = pl.DataFrame(
            {
                "f0": pl.Series("f0", rng.choice(cats, size=512), dtype=pl.Categorical),
                "f1": rng.normal(size=512),
            }
        )
        df_vals = [
            pl.DataFrame(
                {
                    "f0": pl.Series("f0", rng.choice(cats, size=128), dtype=pl.Categorical),
                    "f1": rng.normal(size=128),
                }
            )
            for _ in range(4)
        ]
        y_tr = rng.integers(0, 2, size=512)

        train_dm = xgb.QuantileDMatrix(df_train, y_tr, enable_categorical=True)
        booster = xgb.train(
            {"tree_method": "hist", "objective": "binary:logistic"},
            train_dm,
            num_boost_round=4,
        )

        train_arrow = train_dm.get_categories(export_to_arrow=True).to_arrow()
        for df_v in df_vals:
            val_with_ref = xgb.QuantileDMatrix(df_v, ref=train_dm, enable_categorical=True)
            val_arrow = val_with_ref.get_categories(export_to_arrow=True).to_arrow()
            assert len(val_arrow) == len(train_arrow)
            for (vn, va), (tn, ta) in zip(val_arrow, train_arrow):
                assert vn == tn
                if va is not None:
                    assert va.equals(ta)
            pred_dm = booster.predict(val_with_ref)
            pred_inplace = booster.inplace_predict(df_v)
            assert pred_dm.shape == (128,)
            assert np.isfinite(pred_dm).all()
            np.testing.assert_allclose(pred_dm, pred_inplace, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("dict_size", [1_000, 10_000, 100_000])
def test_categorical_val_dmatrix_shares_ref_cats(
    dict_size: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Correctness + first-call observational timing for the ref-cats alias.

    Builds train and val QuantileDMatrices, trains a booster, and verifies prediction
    parity against inplace_predict on the raw polars DataFrame. Emits per-dict-size
    first-call val construction time via capsys.disabled(); timing is observational
    only.
    """
    rng = np.random.default_rng(2029)
    with pl.StringCache():
        # bind the primer Series to a local so its StringCache entries outlive it
        primer = pl.Series(
            "primer",
            [f"primer_{i}" for i in range(dict_size)],
            dtype=pl.Categorical,
        )
        _unused = primer
        cats = [f"cat_{i:02d}" for i in range(16)]
        train_df = pl.DataFrame(
            {"f0": pl.Series("f0", rng.choice(cats, size=2048), dtype=pl.Categorical)}
        )
        val_df = pl.DataFrame(
            {"f0": pl.Series("f0", rng.choice(cats, size=512), dtype=pl.Categorical)}
        )
        y_train = rng.integers(0, 2, size=2048)

        train_dm = xgb.QuantileDMatrix(train_df, y_train, enable_categorical=True)
        booster = xgb.train(
            {"tree_method": "hist", "objective": "binary:logistic"},
            train_dm,
            num_boost_round=4,
        )

        t0 = time.perf_counter()
        val_dm = xgb.QuantileDMatrix(val_df, ref=train_dm, enable_categorical=True)
        val_t = time.perf_counter() - t0

        assert val_dm.num_row() == 512
        assert val_dm.num_col() == 1

        train_arrow = train_dm.get_categories(export_to_arrow=True).to_arrow()
        val_arrow = val_dm.get_categories(export_to_arrow=True).to_arrow()
        assert len(val_arrow) == len(train_arrow)
        for (vn, va), (tn, ta) in zip(val_arrow, train_arrow):
            assert vn == tn
            if va is not None:
                assert va.equals(ta)

        pred_dm = booster.predict(val_dm)
        pred_inplace = booster.inplace_predict(val_df)
        assert pred_dm.shape == (512,)
        assert np.isfinite(pred_dm).all()
        np.testing.assert_allclose(pred_dm, pred_inplace, rtol=1e-6, atol=1e-7)

    with capsys.disabled():
        print(
            f"[categorical_val_dmatrix_ref_alias_bench] dict_size={dict_size} "
            f"first_call_s={val_t:.4f}"
        )


@pytest.mark.parametrize("primer_size", [1_000, 50_000, 500_000])
def test_categorical_sparse_codes_cpu_bench(
    primer_size: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Times QuantileDMatrix construction under a sparse-codes dictionary.

    Construction is the CPU-side hot path. The wall-clock budget is observational;
    the env-gated assertion catches an O(primer_size) cut-layout regression.
    """
    rng = np.random.default_rng(2031)
    n_real_cats = 16
    n_rows = 2048
    with pl.StringCache():
        primer = pl.Series(
            "primer",
            [f"primer_{i}" for i in range(primer_size)],
            dtype=pl.Categorical,
        )
        _unused = primer
        cats = [f"cat_{i:02d}" for i in range(n_real_cats)]
        df = pl.DataFrame(
            {"f0": pl.Series("f0", rng.choice(cats, size=n_rows), dtype=pl.Categorical)}
        )
        y = rng.integers(0, 2, size=n_rows)

        t0 = time.perf_counter()
        dm = xgb.QuantileDMatrix(df, y, enable_categorical=True)
        construct_t = time.perf_counter() - t0

        assert dm.num_row() == n_rows
        assert dm.num_col() == 1
        # explicit env gate keeps CI green on contended runners; with the gate set, a
        # regression that re-introduces O(primer_size) cut materialisation pushes
        # construct well past 10s on this primer range
        if os.environ.get("XGB_PERF_ASSERT") == "1":
            assert construct_t < 10.0, (
                f"construct_t={construct_t:.3f}s for primer={primer_size}"
            )
        booster = xgb.train(
            {"tree_method": "hist", "objective": "binary:logistic"}, dm, num_boost_round=1
        )
        pred = booster.inplace_predict(df)
        assert np.isfinite(pred).all()
    with capsys.disabled():
        print(
            f"[sparse_codes_cpu_bench] primer_size={primer_size:>7d} "
            f"n_real_cats={n_real_cats} construct_s={construct_t:.4f}"
        )
