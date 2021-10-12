import json
import xgboost as xgb
import pytest
import tempfile
import sys
import numpy as np
import os

sys.path.append("tests/python")
import testing as tm               # noqa
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


def test_boost_from_prediction_gpu_hist():
    twskl.run_boost_from_prediction('gpu_hist')


def test_num_parallel_tree():
    twskl.run_boston_housing_rf_regression("gpu_hist")


@pytest.mark.skipif(**tm.no_pandas())
@pytest.mark.skipif(**tm.no_sklearn())
def test_categorical():
    import pandas as pd
    from sklearn.datasets import load_svmlight_file

    data_dir = os.path.join(tm.PROJECT_ROOT, "demo", "data")
    X, y = load_svmlight_file(os.path.join(data_dir, "agaricus.txt.train"))
    clf = xgb.XGBClassifier(
        tree_method="gpu_hist",
        use_label_encoder=False,
        enable_categorical=True,
        n_estimators=10,
    )
    X = pd.DataFrame(X.todense()).astype("category")
    clf.fit(X, y)
    assert not clf._can_use_inplace_predict()

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
