import json
import os
import pickle
import random
import tempfile
from typing import Callable, Optional

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.ranking import run_ranking_qid_df
from xgboost.testing.shared import get_feature_weights, validate_data_initialization
from xgboost.testing.updater import get_basescore

rng = np.random.RandomState(1994)
pytestmark = [pytest.mark.skipif(**tm.no_sklearn()), tm.timeout(30)]


def test_binary_classification():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    digits = load_digits(n_class=2)
    y = digits['target']
    X = digits['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for cls in (xgb.XGBClassifier, xgb.XGBRFClassifier):
        for train_index, test_index in kf.split(X, y):
            clf = cls(random_state=42)
            xgb_model = clf.fit(X[train_index], y[train_index], eval_metric=['auc', 'logloss'])
            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1


@pytest.mark.parametrize("objective", ["multi:softmax", "multi:softprob"])
def test_multiclass_classification(objective):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import KFold

    def check_pred(preds, labels, output_margin):
        if output_margin:
            err = sum(
                1 for i in range(len(preds)) if preds[i].argmax() != labels[i]
            ) / float(len(preds))
        else:
            err = sum(1 for i in range(len(preds)) if preds[i] != labels[i]) / float(
                len(preds)
            )
        assert err < 0.4

    X, y = load_iris(return_X_y=True)
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBClassifier(objective=objective).fit(
            X[train_index], y[train_index]
        )
        assert xgb_model.get_booster().num_boosted_rounds() == 100
        preds = xgb_model.predict(X[test_index])
        # test other params in XGBClassifier().fit
        preds2 = xgb_model.predict(
            X[test_index], output_margin=True, iteration_range=(0, 1)
        )
        preds3 = xgb_model.predict(
            X[test_index], output_margin=True, iteration_range=None
        )
        preds4 = xgb_model.predict(
            X[test_index], output_margin=False, iteration_range=(0, 1)
        )
        labels = y[test_index]

        check_pred(preds, labels, output_margin=False)
        check_pred(preds2, labels, output_margin=True)
        check_pred(preds3, labels, output_margin=True)
        check_pred(preds4, labels, output_margin=False)

    cls = xgb.XGBClassifier(n_estimators=4).fit(X, y)
    assert cls.n_classes_ == 3
    proba = cls.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] == cls.n_classes_

    # custom objective, the default is multi:softprob so no transformation is required.
    cls = xgb.XGBClassifier(n_estimators=4, objective=tm.softprob_obj(3)).fit(X, y)
    proba = cls.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] == cls.n_classes_


def test_best_iteration():
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    def train(booster: str, forest: Optional[int]) -> None:
        rounds = 4
        cls = xgb.XGBClassifier(
            n_estimators=rounds, num_parallel_tree=forest, booster=booster
        ).fit(
            X, y, eval_set=[(X, y)], early_stopping_rounds=3
        )
        assert cls.best_iteration == rounds - 1

        # best_iteration is used by default, assert that under gblinear it's
        # automatically ignored due to being 0.
        cls.predict(X)

    num_parallel_tree = 4
    train('gbtree', num_parallel_tree)
    train('dart', num_parallel_tree)
    train('gblinear', None)


def test_ranking():
    # generate random data
    x_train = np.random.rand(1000, 10)
    y_train = np.random.randint(5, size=1000)
    train_group = np.repeat(50, 20)

    x_valid = np.random.rand(200, 10)
    y_valid = np.random.randint(5, size=200)
    valid_group = np.repeat(50, 4)

    x_test = np.random.rand(100, 10)

    params = {
        "tree_method": "exact",
        "learning_rate": 0.1,
        "gamma": 1.0,
        "min_child_weight": 0.1,
        "max_depth": 6,
        "eval_metric": "ndcg",
        "n_estimators": 4,
    }
    model = xgb.sklearn.XGBRanker(**params)
    model.fit(
        x_train,
        y_train,
        group=train_group,
        eval_set=[(x_valid, y_valid)],
        eval_group=[valid_group],
    )
    assert model.evals_result()

    pred = model.predict(x_test)

    train_data = xgb.DMatrix(x_train, y_train)
    valid_data = xgb.DMatrix(x_valid, y_valid)
    test_data = xgb.DMatrix(x_test)
    train_data.set_group(train_group)
    assert train_data.get_label().shape[0] == x_train.shape[0]
    valid_data.set_group(valid_group)

    params_orig = {
        "tree_method": "exact",
        "objective": "rank:pairwise",
        "eta": 0.1,
        "gamma": 1.0,
        "min_child_weight": 0.1,
        "max_depth": 6,
        "eval_metric": "ndcg",
    }
    xgb_model_orig = xgb.train(
        params_orig, train_data, num_boost_round=4, evals=[(valid_data, "validation")]
    )
    pred_orig = xgb_model_orig.predict(test_data)

    np.testing.assert_almost_equal(pred, pred_orig)


def test_ranking_metric() -> None:
    from sklearn.metrics import roc_auc_score

    X, y, qid, w = tm.make_ltr(512, 4, 3, 1)
    # use auc for test as ndcg_score in sklearn works only on label gain instead of exp
    # gain.
    # note that the auc in sklearn is different from the one in XGBoost. The one in
    # sklearn compares the number of mis-classified docs, while the one in xgboost
    # compares the number of mis-classified pairs.
    ltr = xgb.XGBRanker(
        eval_metric=roc_auc_score,
        n_estimators=10,
        tree_method="hist",
        max_depth=2,
        objective="rank:pairwise",
    )
    ltr.fit(
        X,
        y,
        qid=qid,
        sample_weight=w,
        eval_set=[(X, y)],
        eval_qid=[qid],
        sample_weight_eval_set=[w],
        verbose=True,
    )
    results = ltr.evals_result()
    assert results["validation_0"]["roc_auc_score"][-1] > 0.6


@pytest.mark.skipif(**tm.no_pandas())
def test_ranking_qid_df():
    import pandas as pd

    run_ranking_qid_df(pd, "hist")


def test_stacking_regression():
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    estimators = [
        ('gbm', xgb.sklearn.XGBRegressor(objective='reg:squarederror')),
        ('lr', RidgeCV())
    ]
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(n_estimators=10,
                                              random_state=42)
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    reg.fit(X_train, y_train).score(X_test, y_test)


def test_stacking_classification():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    X, y = load_iris(return_X_y=True)
    estimators = [
        ('gbm', xgb.sklearn.XGBClassifier()),
        ('svr', make_pipeline(StandardScaler(),
                              LinearSVC(random_state=42)))
    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf.fit(X_train, y_train).score(X_test, y_test)

@pytest.mark.skipif(**tm.no_pandas())
def test_feature_importances_weight():
    from sklearn.datasets import load_digits

    digits = load_digits(n_class=2)
    y = digits["target"]
    X = digits["data"]

    xgb_model = xgb.XGBClassifier(
        random_state=0,
        tree_method="exact",
        learning_rate=0.1,
        importance_type="weight",
        base_score=0.5,
    ).fit(X, y)

    exp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00833333, 0.,
                    0., 0., 0., 0., 0., 0., 0., 0.025, 0.14166667, 0., 0., 0.,
                    0., 0., 0., 0.00833333, 0.25833333, 0., 0., 0., 0.,
                    0.03333334, 0.03333334, 0., 0.32499999, 0., 0., 0., 0.,
                    0.05, 0.06666667, 0., 0., 0., 0., 0., 0., 0., 0.04166667,
                    0., 0., 0., 0., 0., 0., 0., 0.00833333, 0., 0., 0., 0.,
                    0.], dtype=np.float32)

    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)

    # numeric columns
    import pandas as pd
    y = pd.Series(digits['target'])
    X = pd.DataFrame(digits['data'])
    xgb_model = xgb.XGBClassifier(
        random_state=0,
        tree_method="exact",
        learning_rate=0.1,
        base_score=.5,
        importance_type="weight"
    ).fit(X, y)
    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)

    xgb_model = xgb.XGBClassifier(
        random_state=0,
        tree_method="exact",
        learning_rate=0.1,
        importance_type="weight",
        base_score=.5,
    ).fit(X, y)
    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)

    with pytest.raises(ValueError):
        xgb_model.set_params(importance_type="foo")
        xgb_model.feature_importances_

    X, y = load_digits(n_class=3, return_X_y=True)

    cls = xgb.XGBClassifier(booster="gblinear", n_estimators=4)
    cls.fit(X, y)
    assert cls.feature_importances_.shape[0] == X.shape[1]
    assert cls.feature_importances_.shape[1] == 3
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.json")
        cls.save_model(path)
        with open(path, "r") as fd:
            model = json.load(fd)
    weights = np.array(
        model["learner"]["gradient_booster"]["model"]["weights"]
    ).reshape((cls.n_features_in_ + 1, 3))
    weights = weights[:-1, ...]
    np.testing.assert_allclose(
        weights / weights.sum(), cls.feature_importances_, rtol=1e-6
    )

    with pytest.raises(ValueError):
        cls.set_params(importance_type="cover")
        cls.feature_importances_


@pytest.mark.skipif(**tm.no_pandas())
def test_feature_importances_gain():
    from sklearn.datasets import load_digits

    digits = load_digits(n_class=2)
    y = digits['target']
    X = digits['data']
    xgb_model = xgb.XGBClassifier(
        random_state=0, tree_method="exact",
        learning_rate=0.1,
        importance_type="gain",
        base_score=0.5,
    ).fit(X, y)

    exp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0.00326159, 0., 0., 0., 0., 0., 0., 0., 0.,
                    0.00297238, 0.00988034, 0., 0., 0., 0., 0., 0.,
                    0.03512521, 0.41123885, 0., 0., 0., 0.,
                    0.01326332, 0.00160674, 0., 0.4206952, 0., 0., 0.,
                    0., 0.00616747, 0.01237546, 0., 0., 0., 0., 0.,
                    0., 0., 0.08240705, 0., 0., 0., 0., 0., 0., 0.,
                    0.00100649, 0., 0., 0., 0., 0.], dtype=np.float32)

    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)

    # numeric columns
    import pandas as pd
    y = pd.Series(digits['target'])
    X = pd.DataFrame(digits['data'])
    xgb_model = xgb.XGBClassifier(
        random_state=0,
        tree_method="exact",
        learning_rate=0.1,
        importance_type="gain",
        base_score=0.5,
    ).fit(X, y)
    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)

    xgb_model = xgb.XGBClassifier(
        random_state=0,
        tree_method="exact",
        learning_rate=0.1,
        importance_type="gain",
        base_score=0.5,
    ).fit(X, y)
    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)

    # no split can be found
    cls = xgb.XGBClassifier(min_child_weight=1000, tree_method="hist", n_estimators=1)
    cls.fit(X, y)
    assert np.all(cls.feature_importances_ == 0)


def test_select_feature():
    from sklearn.datasets import load_digits
    from sklearn.feature_selection import SelectFromModel
    digits = load_digits(n_class=2)
    y = digits['target']
    X = digits['data']
    cls = xgb.XGBClassifier()
    cls.fit(X, y)
    selector = SelectFromModel(cls, prefit=True, max_features=1)
    X_selected = selector.transform(X)
    assert X_selected.shape[1] == 1


def test_num_parallel_tree():
    from sklearn.datasets import load_diabetes

    reg = xgb.XGBRegressor(n_estimators=4, num_parallel_tree=4, tree_method="hist")
    X, y = load_diabetes(return_X_y=True)
    bst = reg.fit(X=X, y=y)
    dump = bst.get_booster().get_dump(dump_format="json")
    assert len(dump) == 16

    reg = xgb.XGBRFRegressor(n_estimators=4)
    bst = reg.fit(X=X, y=y)
    dump = bst.get_booster().get_dump(dump_format="json")
    assert len(dump) == 4

    config = json.loads(bst.get_booster().save_config())
    assert (
        int(
            config["learner"]["gradient_booster"]["gbtree_model_param"][
                "num_parallel_tree"
            ]
        )
        == 4
    )


def test_regression():
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    X, y = fetch_california_housing(return_X_y=True)
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])

        preds = xgb_model.predict(X[test_index])
        # test other params in XGBRegressor().fit
        preds2 = xgb_model.predict(
            X[test_index], output_margin=True, iteration_range=(0, 3)
        )
        preds3 = xgb_model.predict(
            X[test_index], output_margin=True, iteration_range=None
        )
        preds4 = xgb_model.predict(
            X[test_index], output_margin=False, iteration_range=(0, 3)
        )
        labels = y[test_index]

        assert mean_squared_error(preds, labels) < 25
        assert mean_squared_error(preds2, labels) < 350
        assert mean_squared_error(preds3, labels) < 25
        assert mean_squared_error(preds4, labels) < 350

        with pytest.raises(AttributeError, match="feature_names_in_"):
            xgb_model.feature_names_in_


def run_housing_rf_regression(tree_method):
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    X, y = fetch_california_housing(return_X_y=True)
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBRFRegressor(random_state=42, tree_method=tree_method).fit(
            X[train_index], y[train_index]
        )
        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
        assert mean_squared_error(preds, labels) < 35

    rfreg = xgb.XGBRFRegressor()
    with pytest.raises(NotImplementedError):
        rfreg.fit(X, y, early_stopping_rounds=10)


def test_rf_regression():
    run_housing_rf_regression("hist")


def test_parameter_tuning():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import GridSearchCV

    X, y = fetch_california_housing(return_X_y=True)
    xgb_model = xgb.XGBRegressor(learning_rate=0.1)
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4],
                                   'n_estimators': [50, 200]},
                       cv=2, verbose=1)
    clf.fit(X, y)
    assert clf.best_score_ < 0.7
    assert clf.best_params_ == {'n_estimators': 200, 'max_depth': 4}


def test_regression_with_custom_objective():
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    def objective_ls(y_true, y_pred):
        grad = (y_pred - y_true)
        hess = np.ones(len(y_true))
        return grad, hess

    X, y = fetch_california_housing(return_X_y=True)
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBRegressor(objective=objective_ls).fit(
            X[train_index], y[train_index]
        )
        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
    assert mean_squared_error(preds, labels) < 25

    # Test that the custom objective function is actually used
    class XGBCustomObjectiveException(Exception):
        pass

    def dummy_objective(y_true, y_pred):
        raise XGBCustomObjectiveException()

    xgb_model = xgb.XGBRegressor(objective=dummy_objective)
    np.testing.assert_raises(XGBCustomObjectiveException, xgb_model.fit, X, y)


def test_classification_with_custom_objective():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    def logregobj(y_true, y_pred):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        grad = y_pred - y_true
        hess = y_pred * (1.0 - y_pred)
        return grad, hess

    digits = load_digits(n_class=2)
    y = digits['target']
    X = digits['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBClassifier(objective=logregobj)
        xgb_model.fit(X[train_index], y[train_index])
        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.1

    # Test that the custom objective function is actually used
    class XGBCustomObjectiveException(Exception):
        pass

    def dummy_objective(y_true, y_preds):
        raise XGBCustomObjectiveException()

    xgb_model = xgb.XGBClassifier(objective=dummy_objective)
    np.testing.assert_raises(
        XGBCustomObjectiveException,
        xgb_model.fit,
        X, y
    )

    cls = xgb.XGBClassifier(n_estimators=1)
    cls.fit(X, y)

    is_called = [False]

    def wrapped(y, p):
        is_called[0] = True
        return logregobj(y, p)

    cls.set_params(objective=wrapped)
    cls.predict(X)              # no throw
    cls.fit(X, y)

    assert is_called[0]


def run_sklearn_api(booster, error, n_est):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    tr_d, te_d, tr_l, te_l = train_test_split(iris.data, iris.target,
                                              train_size=120, test_size=0.2)

    classifier = xgb.XGBClassifier(booster=booster, n_estimators=n_est)
    classifier.fit(tr_d, tr_l)

    preds = classifier.predict(te_d)
    labels = te_l
    err = sum([1 for p, l in zip(preds, labels) if p != l]) * 1.0 / len(te_l)
    assert err < error


def test_sklearn_api():
    run_sklearn_api("gbtree", 0.2, 10)
    run_sklearn_api("gblinear", 0.5, 100)


@pytest.mark.skipif(**tm.no_matplotlib())
@pytest.mark.skipif(**tm.no_graphviz())
def test_sklearn_plotting():
    from sklearn.datasets import load_iris

    iris = load_iris()

    classifier = xgb.XGBClassifier()
    classifier.fit(iris.data, iris.target)

    import matplotlib
    matplotlib.use('Agg')

    from graphviz import Source
    from matplotlib.axes import Axes

    ax = xgb.plot_importance(classifier)
    assert isinstance(ax, Axes)
    assert ax.get_title() == 'Feature importance'
    assert ax.get_xlabel() == 'F score'
    assert ax.get_ylabel() == 'Features'
    assert len(ax.patches) == 4

    g = xgb.to_graphviz(classifier, num_trees=0)
    assert isinstance(g, Source)

    ax = xgb.plot_tree(classifier, num_trees=0)
    assert isinstance(ax, Axes)


@pytest.mark.skipif(**tm.no_pandas())
def test_sklearn_nfolds_cv():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold

    digits = load_digits(n_class=3)
    X = digits['data']
    y = digits['target']
    dm = xgb.DMatrix(X, label=y)

    params = {
        'max_depth': 2,
        'eta': 1,
        'verbosity': 0,
        'objective':
        'multi:softprob',
        'num_class': 3
    }

    seed = 2016
    nfolds = 5
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

    cv1 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds,
                 seed=seed, as_pandas=True)
    cv2 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds,
                 folds=skf, seed=seed, as_pandas=True)
    cv3 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds,
                 stratified=True, seed=seed, as_pandas=True)
    assert cv1.shape[0] == cv2.shape[0] and cv2.shape[0] == cv3.shape[0]
    assert cv2.iloc[-1, 0] == cv3.iloc[-1, 0]


@pytest.mark.skipif(**tm.no_pandas())
def test_split_value_histograms():
    from sklearn.datasets import load_digits

    digits_2class = load_digits(n_class=2)

    X = digits_2class["data"]
    y = digits_2class["target"]

    dm = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 6,
        "eta": 0.01,
        "verbosity": 0,
        "objective": "binary:logistic",
        "base_score": 0.5,
    }

    gbdt = xgb.train(params, dm, num_boost_round=10)
    assert gbdt.get_split_value_histogram("not_there", as_pandas=True).shape[0] == 0
    assert gbdt.get_split_value_histogram("not_there", as_pandas=False).shape[0] == 0
    assert gbdt.get_split_value_histogram("f28", bins=0).shape[0] == 1
    assert gbdt.get_split_value_histogram("f28", bins=1).shape[0] == 1
    assert gbdt.get_split_value_histogram("f28", bins=2).shape[0] == 2
    assert gbdt.get_split_value_histogram("f28", bins=5).shape[0] == 2
    assert gbdt.get_split_value_histogram("f28", bins=None).shape[0] == 2


def test_sklearn_random_state():
    clf = xgb.XGBClassifier(random_state=402)
    assert clf.get_xgb_params()['random_state'] == 402

    clf = xgb.XGBClassifier(random_state=401)
    assert clf.get_xgb_params()['random_state'] == 401

    random_state = np.random.RandomState(seed=403)
    clf = xgb.XGBClassifier(random_state=random_state)
    assert isinstance(clf.get_xgb_params()['random_state'], int)


def test_sklearn_n_jobs():
    clf = xgb.XGBClassifier(n_jobs=1)
    assert clf.get_xgb_params()['n_jobs'] == 1

    clf = xgb.XGBClassifier(n_jobs=2)
    assert clf.get_xgb_params()['n_jobs'] == 2


def test_parameters_access():
    from sklearn import datasets

    params = {"updater": "grow_gpu_hist", "subsample": 0.5, "n_jobs": -1}
    clf = xgb.XGBClassifier(n_estimators=1000, **params)
    assert clf.get_params()["updater"] == "grow_gpu_hist"
    assert clf.get_params()["subsample"] == 0.5
    assert clf.get_params()["n_estimators"] == 1000

    clf = xgb.XGBClassifier(n_estimators=1, nthread=4)
    X, y = datasets.load_iris(return_X_y=True)
    clf.fit(X, y)

    config = json.loads(clf.get_booster().save_config())
    assert int(config["learner"]["generic_param"]["nthread"]) == 4

    clf.set_params(nthread=16)
    config = json.loads(clf.get_booster().save_config())
    assert int(config["learner"]["generic_param"]["nthread"]) == 16

    clf.predict(X)
    config = json.loads(clf.get_booster().save_config())
    assert int(config["learner"]["generic_param"]["nthread"]) == 16

    clf = xgb.XGBClassifier(n_estimators=2)
    assert clf.tree_method is None
    assert clf.get_params()["tree_method"] is None
    clf.fit(X, y)
    assert clf.get_params()["tree_method"] is None

    def save_load(clf: xgb.XGBClassifier) -> xgb.XGBClassifier:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            clf.save_model(path)
            clf = xgb.XGBClassifier()
            clf.load_model(path)
        return clf

    def get_tm(clf: xgb.XGBClassifier) -> str:
        tm = json.loads(clf.get_booster().save_config())["learner"]["gradient_booster"][
            "gbtree_train_param"
        ]["tree_method"]
        return tm

    assert get_tm(clf) == "exact"

    clf = pickle.loads(pickle.dumps(clf))

    assert clf.tree_method is None
    assert clf.n_estimators == 2
    assert clf.get_params()["tree_method"] is None
    assert clf.get_params()["n_estimators"] == 2
    assert get_tm(clf) == "exact"  # preserved for pickle

    clf = save_load(clf)

    assert clf.tree_method is None
    assert clf.n_estimators is None
    assert clf.get_params()["tree_method"] is None
    assert clf.get_params()["n_estimators"] is None
    assert get_tm(clf) == "auto"  # discarded for save/load_model

    clf.set_params(tree_method="hist")
    assert clf.get_params()["tree_method"] == "hist"
    clf = pickle.loads(pickle.dumps(clf))
    assert clf.get_params()["tree_method"] == "hist"
    clf = save_load(clf)
    assert clf.get_params()["tree_method"] is None


def test_kwargs_error():
    params = {'updater': 'grow_gpu_hist', 'subsample': .5, 'n_jobs': -1}
    with pytest.raises(TypeError):
        clf = xgb.XGBClassifier(n_jobs=1000, **params)
        assert isinstance(clf, xgb.XGBClassifier)


def test_kwargs_grid_search():
    from sklearn import datasets
    from sklearn.model_selection import GridSearchCV

    params = {'tree_method': 'hist'}
    clf = xgb.XGBClassifier(n_estimators=1, learning_rate=1.0, **params)
    assert clf.get_params()['tree_method'] == 'hist'
    # 'max_leaves' is not a default argument of XGBClassifier
    # Check we can still do grid search over this parameter
    search_params = {'max_leaves': range(2, 5)}
    grid_cv = GridSearchCV(clf, search_params, cv=5)
    iris = datasets.load_iris()
    grid_cv.fit(iris.data, iris.target)

    # Expect unique results for each parameter value
    # This confirms sklearn is able to successfully update the parameter
    means = grid_cv.cv_results_['mean_test_score']
    assert len(means) == len(set(means))


def test_sklearn_clone():
    from sklearn.base import clone

    clf = xgb.XGBClassifier(n_jobs=2)
    clf.n_jobs = -1
    clone(clf)


def test_sklearn_get_default_params():
    from sklearn.datasets import load_digits

    digits_2class = load_digits(n_class=2)
    X = digits_2class["data"]
    y = digits_2class["target"]
    cls = xgb.XGBClassifier()
    assert cls.get_params()["base_score"] is None
    cls.fit(X[:4, ...], y[:4, ...])
    base_score = get_basescore(cls)
    np.testing.assert_equal(base_score, 0.5)


def run_validation_weights(model):
    from sklearn.datasets import make_hastie_10_2

    # prepare training and test data
    X, y = make_hastie_10_2(n_samples=2000, random_state=42)
    labels, y = np.unique(y, return_inverse=True)
    X_train, X_test = X[:1600], X[1600:]
    y_train, y_test = y[:1600], y[1600:]

    # instantiate model
    param_dist = {'objective': 'binary:logistic', 'n_estimators': 2,
                  'random_state': 123}
    clf = model(**param_dist)

    # train it using instance weights only in the training set
    weights_train = np.random.choice([1, 2], len(X_train))
    clf.fit(X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_metric='logloss',
            verbose=False)

    # evaluate logloss metric on test set *without* using weights
    evals_result_without_weights = clf.evals_result()
    logloss_without_weights = evals_result_without_weights[
        "validation_0"]["logloss"]

    # now use weights for the test set
    np.random.seed(0)
    weights_test = np.random.choice([1, 2], len(X_test))
    clf.fit(X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            sample_weight_eval_set=[weights_test],
            eval_metric='logloss',
            verbose=False)
    evals_result_with_weights = clf.evals_result()
    logloss_with_weights = evals_result_with_weights["validation_0"]["logloss"]

    # check that the logloss in the test set is actually different when using
    # weights than when not using them
    assert all((logloss_with_weights[i] != logloss_without_weights[i]
                for i in [0, 1]))

    with pytest.raises(ValueError):
        # length of eval set and sample weight doesn't match.
        clf.fit(X_train, y_train, sample_weight=weights_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                sample_weight_eval_set=[weights_train])

    with pytest.raises(ValueError):
        cls = xgb.XGBClassifier()
        cls.fit(X_train, y_train, sample_weight=weights_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                sample_weight_eval_set=[weights_train])


def test_validation_weights():
    run_validation_weights(xgb.XGBModel)
    run_validation_weights(xgb.XGBClassifier)


def save_load_model(model_path):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    digits = load_digits(n_class=2)
    y = digits['target']
    X = digits['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
        xgb_model.save_model(model_path)

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_path)

        assert isinstance(xgb_model.classes_, np.ndarray)
        np.testing.assert_equal(xgb_model.classes_, np.array([0, 1]))
        assert isinstance(xgb_model._Booster, xgb.Booster)

        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.1
        assert xgb_model.get_booster().attr('scikit_learn') is None

        # test native booster
        preds = xgb_model.predict(X[test_index], output_margin=True)
        booster = xgb.Booster(model_file=model_path)
        predt_1 = booster.predict(xgb.DMatrix(X[test_index]),
                                  output_margin=True)
        assert np.allclose(preds, predt_1)

        with pytest.raises(TypeError):
            xgb_model = xgb.XGBModel()
            xgb_model.load_model(model_path)


def test_save_load_model():
    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, 'digits.model')
        save_load_model(model_path)

    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, 'digits.model.json')
        save_load_model(model_path)

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, 'digits.model.ubj')
        digits = load_digits(n_class=2)
        y = digits['target']
        X = digits['data']
        booster = xgb.train({'tree_method': 'hist',
                             'objective': 'binary:logistic'},
                            dtrain=xgb.DMatrix(X, y),
                            num_boost_round=4)
        predt_0 = booster.predict(xgb.DMatrix(X))
        booster.save_model(model_path)
        cls = xgb.XGBClassifier()
        cls.load_model(model_path)

        proba = cls.predict_proba(X)
        assert proba.shape[0] == X.shape[0]
        assert proba.shape[1] == 2  # binary

        predt_1 = cls.predict_proba(X)[:, 1]
        assert np.allclose(predt_0, predt_1)

        cls = xgb.XGBModel()
        cls.load_model(model_path)
        predt_1 = cls.predict(X)
        assert np.allclose(predt_0, predt_1)

        # mclass
        X, y = load_digits(n_class=10, return_X_y=True)
        # small test_size to force early stop
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.01, random_state=1
        )
        clf = xgb.XGBClassifier(
            n_estimators=64, tree_method="hist", early_stopping_rounds=2
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        score = clf.best_score
        clf.save_model(model_path)

        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        assert clf.classes_.size == 10
        np.testing.assert_equal(clf.classes_, np.arange(10))
        assert clf.n_classes_ == 10

        assert clf.best_iteration == 27
        assert clf.best_score == score


def test_RFECV():
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
    from sklearn.feature_selection import RFECV

    # Regression
    X, y = load_diabetes(return_X_y=True)
    bst = xgb.XGBRegressor(booster='gblinear', learning_rate=0.1,
                           n_estimators=10,
                           objective='reg:squarederror',
                           random_state=0, verbosity=0)
    rfecv = RFECV(
        estimator=bst, step=1, cv=3, scoring='neg_mean_squared_error')
    rfecv.fit(X, y)

    # Binary classification
    X, y = load_breast_cancer(return_X_y=True)
    bst = xgb.XGBClassifier(booster='gblinear', learning_rate=0.1,
                            n_estimators=10,
                            objective='binary:logistic',
                            random_state=0, verbosity=0)
    rfecv = RFECV(estimator=bst, step=0.5, cv=3, scoring='roc_auc')
    rfecv.fit(X, y)

    # Multi-class classification
    X, y = load_iris(return_X_y=True)
    bst = xgb.XGBClassifier(base_score=0.4, booster='gblinear',
                            learning_rate=0.1,
                            n_estimators=10,
                            objective='multi:softprob',
                            random_state=0, reg_alpha=0.001, reg_lambda=0.01,
                            scale_pos_weight=0.5, verbosity=0)
    rfecv = RFECV(estimator=bst, step=0.5, cv=3, scoring='neg_log_loss')
    rfecv.fit(X, y)

    X[0:4, :] = np.nan          # verify scikit_learn doesn't throw with nan
    reg = xgb.XGBRegressor()
    rfecv = RFECV(estimator=reg)
    rfecv.fit(X, y)

    cls = xgb.XGBClassifier()
    rfecv = RFECV(estimator=cls, step=0.5, cv=3,
                  scoring='neg_mean_squared_error')
    rfecv.fit(X, y)


def test_XGBClassifier_resume():
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import log_loss

    with tempfile.TemporaryDirectory() as tempdir:
        model1_path = os.path.join(tempdir, 'test_XGBClassifier.model')
        model1_booster_path = os.path.join(tempdir, 'test_XGBClassifier.booster')

        X, Y = load_breast_cancer(return_X_y=True)

        model1 = xgb.XGBClassifier(
            learning_rate=0.3, random_state=0, n_estimators=8)
        model1.fit(X, Y)

        pred1 = model1.predict(X)
        log_loss1 = log_loss(pred1, Y)

        # file name of stored xgb model
        model1.save_model(model1_path)
        model2 = xgb.XGBClassifier(
            learning_rate=0.3, random_state=0, n_estimators=8)
        model2.fit(X, Y, xgb_model=model1_path)

        pred2 = model2.predict(X)
        log_loss2 = log_loss(pred2, Y)

        assert np.any(pred1 != pred2)
        assert log_loss1 > log_loss2

        # file name of 'Booster' instance Xgb model
        model1.get_booster().save_model(model1_booster_path)
        model2 = xgb.XGBClassifier(
            learning_rate=0.3, random_state=0, n_estimators=8)
        model2.fit(X, Y, xgb_model=model1_booster_path)

        pred2 = model2.predict(X)
        log_loss2 = log_loss(pred2, Y)

        assert np.any(pred1 != pred2)
        assert log_loss1 > log_loss2


def test_constraint_parameters():
    reg = xgb.XGBRegressor(interaction_constraints="[[0, 1], [2, 3, 4]]")
    X = np.random.randn(10, 10)
    y = np.random.randn(10)
    reg.fit(X, y)

    config = json.loads(reg.get_booster().save_config())
    assert (
        config["learner"]["gradient_booster"]["tree_train_param"][
            "interaction_constraints"
        ]
        == "[[0, 1], [2, 3, 4]]"
    )


def test_parameter_validation():
    reg = xgb.XGBRegressor(foo='bar', verbosity=1)
    X = np.random.randn(10, 10)
    y = np.random.randn(10)
    with tm.captured_output() as (out, err):
        reg.fit(X, y)
        output = out.getvalue().strip()

    assert output.find('foo') != -1

    reg = xgb.XGBRegressor(n_estimators=2, missing=3,
                           importance_type='gain', verbosity=1)
    X = np.random.randn(10, 10)
    y = np.random.randn(10)
    with tm.captured_output() as (out, err):
        reg.fit(X, y)
        output = out.getvalue().strip()

    assert len(output) == 0


def test_deprecate_position_arg():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True, n_class=2)
    w = y
    with pytest.warns(FutureWarning):
        xgb.XGBRegressor(3, learning_rate=0.1)
    model = xgb.XGBRegressor(n_estimators=1)
    with pytest.warns(FutureWarning):
        model.fit(X, y, w)

    with pytest.warns(FutureWarning):
        xgb.XGBClassifier(1)
    model = xgb.XGBClassifier(n_estimators=1)
    with pytest.warns(FutureWarning):
        model.fit(X, y, w)

    with pytest.warns(FutureWarning):
        xgb.XGBRanker('rank:ndcg', learning_rate=0.1)
    model = xgb.XGBRanker(n_estimators=1)
    group = np.repeat(1, X.shape[0])
    with pytest.warns(FutureWarning):
        model.fit(X, y, group)

    with pytest.warns(FutureWarning):
        xgb.XGBRFRegressor(1, learning_rate=0.1)
    model = xgb.XGBRFRegressor(n_estimators=1)
    with pytest.warns(FutureWarning):
        model.fit(X, y, w)

    model = xgb.XGBRFClassifier(n_estimators=1)
    with pytest.warns(FutureWarning):
        model.fit(X, y, w)


@pytest.mark.skipif(**tm.no_pandas())
def test_pandas_input():
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV

    rng = np.random.RandomState(1994)

    kRows = 100
    kCols = 6

    X = rng.randint(low=0, high=2, size=kRows * kCols)
    X = X.reshape(kRows, kCols)

    df = pd.DataFrame(X)
    feature_names = []
    for i in range(1, kCols):
        feature_names += ["k" + str(i)]

    df.columns = ["status"] + feature_names

    target = df["status"]
    train = df.drop(columns=["status"])
    model = xgb.XGBClassifier()
    model.fit(train, target)
    np.testing.assert_equal(model.feature_names_in_, np.array(feature_names))

    columns = list(train.columns)
    random.shuffle(columns, lambda: 0.1)
    df_incorrect = df[columns]
    with pytest.raises(ValueError):
        model.predict(df_incorrect)

    clf_isotonic = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    clf_isotonic.fit(train, target)
    assert isinstance(
        clf_isotonic.calibrated_classifiers_[0].estimator, xgb.XGBClassifier
    )
    np.testing.assert_allclose(np.array(clf_isotonic.classes_), np.array([0, 1]))

    train_ser = train["k1"]
    assert isinstance(train_ser, pd.Series)
    model = xgb.XGBClassifier(n_estimators=8)
    model.fit(train_ser, target, eval_set=[(train_ser, target)])
    assert tm.non_increasing(model.evals_result()["validation_0"]["logloss"])


@pytest.mark.parametrize("tree_method", ["approx", "hist"])
def test_feature_weights(tree_method):
    kRows = 512
    kCols = 64
    X = rng.randn(kRows, kCols)
    y = rng.randn(kRows)

    fw = np.ones(shape=(kCols,))
    for i in range(kCols):
        fw[i] *= float(i)

    parser_path = os.path.join(tm.demo_dir(__file__), "json-model", "json_parser.py")
    poly_increasing = get_feature_weights(
        X, y, fw, parser_path, tree_method, xgb.XGBRegressor
    )

    fw = np.ones(shape=(kCols,))
    for i in range(kCols):
        fw[i] *= float(kCols - i)
    poly_decreasing = get_feature_weights(
        X, y, fw, parser_path, tree_method, xgb.XGBRegressor
    )

    # Approxmated test, this is dependent on the implementation of random
    # number generator in std library.
    assert poly_increasing[0] > 0.08
    assert poly_decreasing[0] < -0.08


def run_boost_from_prediction_binary(tree_method, X, y, as_frame: Optional[Callable]):
    """
    Parameters
    ----------

    as_frame: A callable function to convert margin into DataFrame, useful for different
    df implementations.
    """

    model_0 = xgb.XGBClassifier(
        learning_rate=0.3, random_state=0, n_estimators=4, tree_method=tree_method
    )
    model_0.fit(X=X, y=y)
    margin = model_0.predict(X, output_margin=True)
    if as_frame is not None:
        margin = as_frame(margin)

    model_1 = xgb.XGBClassifier(
        learning_rate=0.3, random_state=0, n_estimators=4, tree_method=tree_method
    )
    model_1.fit(X=X, y=y, base_margin=margin)
    predictions_1 = model_1.predict(X, base_margin=margin)

    cls_2 = xgb.XGBClassifier(
        learning_rate=0.3, random_state=0, n_estimators=8, tree_method=tree_method
    )
    cls_2.fit(X=X, y=y)
    predictions_2 = cls_2.predict(X)
    np.testing.assert_allclose(predictions_1, predictions_2)


def run_boost_from_prediction_multi_clasas(
    estimator, tree_method, X, y, as_frame: Optional[Callable]
):
    # Multi-class
    model_0 = estimator(
        learning_rate=0.3, random_state=0, n_estimators=4, tree_method=tree_method
    )
    model_0.fit(X=X, y=y)
    margin = model_0.get_booster().inplace_predict(X, predict_type="margin")
    if as_frame is not None:
        margin = as_frame(margin)

    model_1 = estimator(
        learning_rate=0.3, random_state=0, n_estimators=4, tree_method=tree_method
    )
    model_1.fit(X=X, y=y, base_margin=margin)
    predictions_1 = model_1.get_booster().predict(
        xgb.DMatrix(X, base_margin=margin), output_margin=True
    )

    model_2 = estimator(
        learning_rate=0.3, random_state=0, n_estimators=8, tree_method=tree_method
    )
    model_2.fit(X=X, y=y)
    predictions_2 = model_2.get_booster().inplace_predict(X, predict_type="margin")

    if hasattr(predictions_1, "get"):
        predictions_1 = predictions_1.get()
    if hasattr(predictions_2, "get"):
        predictions_2 = predictions_2.get()
    np.testing.assert_allclose(predictions_1, predictions_2, atol=1e-6)


@pytest.mark.parametrize("tree_method", ["hist", "approx", "exact"])
def test_boost_from_prediction(tree_method):
    import pandas as pd
    from sklearn.datasets import load_breast_cancer, load_iris, make_regression

    X, y = load_breast_cancer(return_X_y=True)

    run_boost_from_prediction_binary(tree_method, X, y, None)
    run_boost_from_prediction_binary(tree_method, X, y, pd.DataFrame)

    X, y = load_iris(return_X_y=True)

    run_boost_from_prediction_multi_clasas(xgb.XGBClassifier, tree_method, X, y, None)
    run_boost_from_prediction_multi_clasas(
        xgb.XGBClassifier, tree_method, X, y, pd.DataFrame
    )

    X, y = make_regression(n_samples=100, n_targets=4)
    run_boost_from_prediction_multi_clasas(xgb.XGBRegressor, tree_method, X, y, None)


def test_estimator_type():
    assert xgb.XGBClassifier._estimator_type == "classifier"
    assert xgb.XGBRFClassifier._estimator_type == "classifier"
    assert xgb.XGBRegressor._estimator_type == "regressor"
    assert xgb.XGBRFRegressor._estimator_type == "regressor"
    assert xgb.XGBRanker._estimator_type == "ranker"

    from sklearn.datasets import load_digits

    X, y = load_digits(n_class=2, return_X_y=True)
    cls = xgb.XGBClassifier(n_estimators=2).fit(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cls.json")
        cls.save_model(path)

        reg = xgb.XGBRegressor()
        with pytest.raises(TypeError):
            reg.load_model(path)

        cls = xgb.XGBClassifier()
        cls.load_model(path)  # no error


def test_multilabel_classification() -> None:
    from sklearn.datasets import make_multilabel_classification

    X, y = make_multilabel_classification(
        n_samples=32, n_classes=5, n_labels=3, random_state=0
    )
    clf = xgb.XGBClassifier(tree_method="hist")
    clf.fit(X, y)
    booster = clf.get_booster()
    learner = json.loads(booster.save_config())["learner"]
    assert int(learner["learner_model_param"]["num_target"]) == 5

    np.testing.assert_allclose(clf.predict(X), y)
    predt = (clf.predict_proba(X) > 0.5).astype(np.int64)
    np.testing.assert_allclose(clf.predict(X), predt)
    assert predt.dtype == np.int64

    y = y.tolist()
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), predt)


def test_data_initialization():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    validate_data_initialization(xgb.DMatrix, xgb.XGBClassifier, X, y)


@parametrize_with_checks([xgb.XGBRegressor()])
def test_estimator_reg(estimator, check):
    if os.environ["PYTEST_CURRENT_TEST"].find("check_supervised_y_no_nan") != -1:
        # The test uses float64 and requires the error message to contain:
        #
        #   "value too large for dtype(float64)",
        #
        # while XGBoost stores values as float32.  But XGBoost does verify the label
        # internally, so we replace this test with custom check.
        rng = np.random.RandomState(888)
        X = rng.randn(10, 5)
        y = np.full(10, np.inf)
        with pytest.raises(
            ValueError, match="contains NaN, infinity or a value too large"
        ):
            estimator.fit(X, y)
        return
    if os.environ["PYTEST_CURRENT_TEST"].find("check_estimators_overwrite_params") != -1:
        # A hack to pass the scikit-learn parameter mutation tests.  XGBoost regressor
        # returns actual internal default values for parameters in `get_params`, but those
        # are set as `None` in sklearn interface to avoid duplication.  So we fit a dummy
        # model and obtain the default parameters here for the mutation tests.
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=2, n_features=1)
        estimator.set_params(**xgb.XGBRegressor().fit(X, y).get_params())

    check(estimator)


def test_categorical():
    X, y = tm.make_categorical(n_samples=32, n_features=2, n_categories=3, onehot=False)
    ft = ["c"] * X.shape[1]
    reg = xgb.XGBRegressor(
        tree_method="hist",
        feature_types=ft,
        max_cat_to_onehot=1,
        enable_categorical=True,
    )
    reg.fit(X.values, y, eval_set=[(X.values, y)])
    from_cat = reg.evals_result()["validation_0"]["rmse"]
    predt_cat = reg.predict(X.values)
    assert reg.get_booster().feature_types == ft
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.json")
        reg.save_model(path)
        reg = xgb.XGBRegressor()
        reg.load_model(path)
        assert reg.feature_types == ft

    onehot, y = tm.make_categorical(
        n_samples=32, n_features=2, n_categories=3, onehot=True
    )
    reg = xgb.XGBRegressor(tree_method="hist")
    reg.fit(onehot, y, eval_set=[(onehot, y)])
    from_enc = reg.evals_result()["validation_0"]["rmse"]
    predt_enc = reg.predict(onehot)

    np.testing.assert_allclose(from_cat, from_enc)
    np.testing.assert_allclose(predt_cat, predt_enc)


def test_prediction_config():
    reg = xgb.XGBRegressor()
    assert reg._can_use_inplace_predict() is True

    reg.set_params(predictor="cpu_predictor")
    assert reg._can_use_inplace_predict() is False

    reg.set_params(predictor="auto")
    assert reg._can_use_inplace_predict() is True

    reg.set_params(predictor=None)
    assert reg._can_use_inplace_predict() is True

    reg.set_params(booster="gblinear")
    assert reg._can_use_inplace_predict() is False


def test_evaluation_metric():
    from sklearn.datasets import load_diabetes, load_digits
    from sklearn.metrics import mean_absolute_error
    X, y = load_diabetes(return_X_y=True)
    n_estimators = 16

    with tm.captured_output() as (out, err):
        reg = xgb.XGBRegressor(
            tree_method="hist",
            eval_metric=mean_absolute_error,
            n_estimators=n_estimators,
        )
        reg.fit(X, y, eval_set=[(X, y)])
        lines = out.getvalue().strip().split('\n')

    assert len(lines) == n_estimators
    for line in lines:
        assert line.find("mean_absolute_error") != -1

    def metric(predt: np.ndarray, Xy: xgb.DMatrix):
        y = Xy.get_label()
        return "m", np.abs(predt - y).sum()

    with pytest.warns(UserWarning):
        reg = xgb.XGBRegressor(
            tree_method="hist",
            n_estimators=1,
        )
        reg.fit(X, y, eval_set=[(X, y)], eval_metric=metric)

    def merror(y_true: np.ndarray, predt: np.ndarray):
        n_samples = y_true.shape[0]
        assert n_samples == predt.size
        errors = np.zeros(y_true.shape[0])
        errors[y != predt] = 1.0
        return np.sum(errors) / n_samples

    X, y = load_digits(n_class=10, return_X_y=True)

    clf = xgb.XGBClassifier(
        tree_method="hist",
        eval_metric=merror,
        n_estimators=16,
        objective="multi:softmax"
    )
    clf.fit(X, y, eval_set=[(X, y)])
    custom = clf.evals_result()

    clf = xgb.XGBClassifier(
        tree_method="hist",
        eval_metric="merror",
        n_estimators=16,
        objective="multi:softmax"
    )
    clf.fit(X, y, eval_set=[(X, y)])
    internal = clf.evals_result()

    np.testing.assert_allclose(
        custom["validation_0"]["merror"],
        internal["validation_0"]["merror"],
        atol=1e-6
    )

    clf = xgb.XGBRFClassifier(
        tree_method="hist", n_estimators=16,
        objective=tm.softprob_obj(10),
        eval_metric=merror,
    )
    with pytest.raises(AssertionError):
        # shape check inside the `merror` function
        clf.fit(X, y, eval_set=[(X, y)])

def test_weighted_evaluation_metric():
    from sklearn.datasets import make_hastie_10_2
    from sklearn.metrics import log_loss
    X, y = make_hastie_10_2(n_samples=2000, random_state=42)
    labels, y = np.unique(y, return_inverse=True)
    X_train, X_test = X[:1600], X[1600:]
    y_train, y_test = y[:1600], y[1600:]
    weights_eval_set = np.random.choice([1, 2], len(X_test))

    np.random.seed(0)
    weights_train = np.random.choice([1, 2], len(X_train))

    clf = xgb.XGBClassifier(
        tree_method="hist",
        eval_metric=log_loss,
        n_estimators=16,
        objective="binary:logistic",
    )
    clf.fit(X_train, y_train, sample_weight=weights_train, eval_set=[(X_test, y_test)],
            sample_weight_eval_set=[weights_eval_set])
    custom = clf.evals_result()

    clf = xgb.XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        n_estimators=16,
        objective="binary:logistic"
    )
    clf.fit(X_train, y_train, sample_weight=weights_train, eval_set=[(X_test, y_test)],
            sample_weight_eval_set=[weights_eval_set])
    internal = clf.evals_result()

    np.testing.assert_allclose(
        custom["validation_0"]["log_loss"],
        internal["validation_0"]["logloss"],
        atol=1e-6
    )
