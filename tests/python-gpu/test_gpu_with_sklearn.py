import json
import os
import sys
import tempfile

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.ranking import run_ranking_qid_df

sys.path.append("tests/python")
import test_with_sklearn as twskl  # noqa

pytestmark = pytest.mark.skipif(**tm.no_sklearn())

rng = np.random.RandomState(1994)


def test_gpu_binary_classification():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    digits = load_digits(n_class=2)
    y = digits['target']
    X = digits['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for cls in (xgb.XGBClassifier, xgb.XGBRFClassifier):
        for train_index, test_index in kf.split(X, y):
            xgb_model = cls(
                random_state=42, tree_method='gpu_hist',
                n_estimators=4, gpu_id='0').fit(X[train_index], y[train_index])
            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_cudf())
def test_boost_from_prediction_gpu_hist():
    import cudf
    import cupy as cp
    from sklearn.datasets import load_breast_cancer, load_digits

    tree_method = "gpu_hist"
    X, y = load_breast_cancer(return_X_y=True)
    X, y = cp.array(X), cp.array(y)

    twskl.run_boost_from_prediction_binary(tree_method, X, y, None)
    twskl.run_boost_from_prediction_binary(tree_method, X, y, cudf.DataFrame)

    X, y = load_digits(return_X_y=True)
    X, y = cp.array(X), cp.array(y)

    twskl.run_boost_from_prediction_multi_clasas(
        xgb.XGBClassifier, tree_method, X, y, None
    )
    twskl.run_boost_from_prediction_multi_clasas(
        xgb.XGBClassifier, tree_method, X, y, cudf.DataFrame
    )


def test_num_parallel_tree():
    twskl.run_housing_rf_regression("gpu_hist")


@pytest.mark.skipif(**tm.no_pandas())
@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_sklearn())
def test_categorical():
    import cudf
    import cupy as cp
    import pandas as pd
    from sklearn.datasets import load_svmlight_file

    data_dir = tm.data_dir(__file__)
    X, y = load_svmlight_file(os.path.join(data_dir, "agaricus.txt.train"))
    clf = xgb.XGBClassifier(
        tree_method="gpu_hist",
        enable_categorical=True,
        n_estimators=10,
    )
    X = pd.DataFrame(X.todense()).astype("category")
    clf.fit(X, y)

    with tempfile.TemporaryDirectory() as tempdir:
        model = os.path.join(tempdir, "categorial.json")
        clf.save_model(model)

        with open(model) as fd:
            categorical = json.load(fd)
            categories_sizes = np.array(
                categorical["learner"]["gradient_booster"]["model"]["trees"][0][
                    "categories_sizes"
                ]
            )
            assert categories_sizes.shape[0] != 0
            np.testing.assert_allclose(categories_sizes, 1)

    def check_predt(X, y):
        reg = xgb.XGBRegressor(
            tree_method="gpu_hist", enable_categorical=True, n_estimators=64
        )
        reg.fit(X, y)
        predts = reg.predict(X)
        booster = reg.get_booster()
        assert "c" in booster.feature_types
        assert len(booster.feature_types) == 1
        inp_predts = booster.inplace_predict(X)
        if isinstance(inp_predts, cp.ndarray):
            inp_predts = cp.asnumpy(inp_predts)
        np.testing.assert_allclose(predts, inp_predts)

    y = [1, 2, 3]
    X = pd.DataFrame({"f0": ["a", "b", "c"]})
    X["f0"] = X["f0"].astype("category")
    check_predt(X, y)

    X = cudf.DataFrame(X)
    check_predt(X, y)


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_cudf())
def test_classififer():
    import cudf
    import cupy as cp
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    y *= 10

    clf = xgb.XGBClassifier(tree_method="gpu_hist", n_estimators=1)

    # numpy
    with pytest.raises(ValueError, match=r"Invalid classes.*"):
        clf.fit(X, y)

    # cupy
    X, y = cp.array(X), cp.array(y)
    with pytest.raises(ValueError, match=r"Invalid classes.*"):
        clf.fit(X, y)

    # cudf
    X, y = cudf.DataFrame(X), cudf.DataFrame(y)
    with pytest.raises(ValueError, match=r"Invalid classes.*"):
        clf.fit(X, y)

    # pandas
    X, y = load_digits(return_X_y=True, as_frame=True)
    y *= 10
    with pytest.raises(ValueError, match=r"Invalid classes.*"):
        clf.fit(X, y)


@pytest.mark.skipif(**tm.no_pandas())
def test_ranking_qid_df():
    import cudf

    run_ranking_qid_df(cudf, "gpu_hist")
