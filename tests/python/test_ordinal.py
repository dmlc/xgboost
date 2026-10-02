import json
from typing import Literal, cast

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost._data_utils import ArrayInf, from_array_interface, pd_cat_inf
from xgboost.testing.ordinal import (
    run_cat_container,
    run_cat_container_iter,
    run_cat_container_mixed,
    run_cat_invalid,
    run_cat_leaf,
    run_cat_oov_in_range,
    run_cat_predict,
    run_cat_shap,
    run_cat_thread_safety,
    run_recode_dmatrix,
    run_recode_dmatrix_predict,
    run_specified_cat,
    run_training_continuation,
    run_update,
    run_validation,
)

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_arrow(), tm.no_pandas()))


def test_cat_container() -> None:
    run_cat_container("cpu")


def test_cat_container_model_slice() -> None:
    import pandas as pd

    X = pd.DataFrame(
        {"cat": pd.Categorical(["a", "b", "a", "c"]), "num": [0.0, 1.0, 2.0, 3.0]}
    )
    Xy = xgb.DMatrix(X, label=np.arange(X.shape[0]), enable_categorical=True)
    booster = xgb.train({"tree_method": "hist"}, Xy, num_boost_round=2)

    expected = booster.get_categories(export_to_arrow=True).to_arrow()
    actual = booster[:1].get_categories(export_to_arrow=True).to_arrow()
    assert actual == expected


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.parametrize("export_to_arrow", [False, True])
def test_training_continuation_empty_categories(
    tree_method: str, export_to_arrow: bool
) -> None:
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "cat": rng.integers(0, 6, 32).astype(np.float32),
            "num": rng.normal(size=32).astype(np.float32),
        }
    )
    y = (X["cat"] < 3).astype(np.float32)
    feature_types = ["c", "q"]
    model = xgb.XGBRegressor(
        tree_method=tree_method,
        n_estimators=2,
        enable_categorical=True,
        feature_types=feature_types,
    )
    model.fit(X, y)
    booster = model.get_booster()
    saved = json.loads(booster.save_raw(raw_format="json"))
    # A saved model can contain numerical encoding columns with no category values.
    saved["learner"]["gradient_booster"]["model"]["cats"] = {
        "enc": [{"offsets": [], "values": []} for _ in X.columns],
        "feature_segments": [0] * (X.shape[1] + 1),
        "sorted_idx": [],
    }
    booster.load_model(bytearray(json.dumps(saved), "utf-8"))

    categories = booster.get_categories(export_to_arrow=export_to_arrow)
    assert categories.empty()
    if export_to_arrow:
        assert categories.to_arrow() == [(name, None) for name in X.columns]

    model.fit(X, y, xgb_model=booster)
    assert model.get_booster().feature_types == feature_types
    assert model.get_booster().num_boosted_rounds() == 4
    continued = json.loads(model.get_booster().save_raw(raw_format="json"))
    trees = continued["learner"]["gradient_booster"]["model"]["trees"]
    assert all(1 in tree["split_type"] for tree in trees[2:])


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("Int64", [-2, 1]),
        ("UInt16", [1, np.iinfo(np.uint16).max]),
        ("UInt32", [1, np.iinfo(np.uint32).max]),
        ("UInt64", [1, np.iinfo(np.int64).max]),
    ],
)
def test_pd_cat_nullable_integer(
    dtype: Literal["Int64", "UInt16", "UInt32", "UInt64"], values: list[int]
) -> None:
    import pandas as pd

    categories = pd.Index(pd.array(values, dtype=dtype))
    names, codes, temporary = pd_cat_inf(
        categories, pd.Series([0, -1, 1], dtype=np.int8)
    )
    expected_dtype = np.dtype(dtype.lower())
    converted = temporary[0]
    assert converted.dtype == expected_dtype
    assert converted.flags.c_contiguous and converted.flags.aligned
    assert "typestr" in names
    numeric_names = cast(ArrayInf, names)
    assert numeric_names["typestr"] == expected_dtype.str
    np.testing.assert_array_equal(from_array_interface(numeric_names), values)
    code_values = np.asarray(from_array_interface(codes))
    assert code_values[0] == 0 and np.isnan(code_values[1]) and code_values[2] == 1


def test_pd_cat_nullable_float() -> None:
    import pandas as pd

    categories = pd.Index(pd.array([1.0, 2.0], dtype="Float64"))
    cat = pd.Categorical.from_codes([0, 1], categories=categories)
    with pytest.raises(xgb.core.XGBoostError, match="floating point dtype"):
        xgb.DMatrix(pd.DataFrame({"c": cat}), enable_categorical=True)


def test_cat_container_mixed() -> None:
    run_cat_container_mixed("cpu")


def test_cat_container_iter() -> None:
    run_cat_container_iter("cpu")


def test_cat_predict() -> None:
    run_cat_predict("cpu")


def test_cat_invalid() -> None:
    run_cat_invalid("cpu")


def test_cat_oov_in_range() -> None:
    run_cat_oov_in_range("cpu")


def test_cat_thread_safety() -> None:
    run_cat_thread_safety("cpu")


def test_cat_shap() -> None:
    run_cat_shap("cpu")


def test_cat_leaf() -> None:
    run_cat_leaf("cpu")


def test_specified_cat() -> None:
    run_specified_cat("cpu")


def test_validation() -> None:
    run_validation("cpu")


def test_recode_dmatrix() -> None:
    run_recode_dmatrix("cpu")


def test_training_continuation() -> None:
    run_training_continuation("cpu")


def test_update() -> None:
    run_update("cpu")


def test_recode_dmatrix_predict() -> None:
    run_recode_dmatrix_predict("cpu")
