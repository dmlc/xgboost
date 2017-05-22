import numpy as np
import random
import xgboost as xgb
import testing as tm

rng = np.random.RandomState(1994)


def test_binary_classification():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits
    try:
        from sklearn.model_selection import KFold
    except:
        from sklearn.cross_validation import KFold

    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    try:
        kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
    except TypeError:  # sklearn.model_selection.KFold uses n_split
        kf = KFold(
            n_splits=2, shuffle=True, random_state=rng
        ).split(np.arange(y.shape[0]))
    for train_index, test_index in kf:
        xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.1


def test_multiclass_classification():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_iris
    try:
        from sklearn.cross_validation import KFold
    except:
        from sklearn.model_selection import KFold

    def check_pred(preds, labels):
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.4

    iris = load_iris()
    y = iris['target']
    X = iris['data']
    kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf:
        xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
        preds = xgb_model.predict(X[test_index])
        # test other params in XGBClassifier().fit
        preds2 = xgb_model.predict(X[test_index], output_margin=True, ntree_limit=3)
        preds3 = xgb_model.predict(X[test_index], output_margin=True, ntree_limit=0)
        preds4 = xgb_model.predict(X[test_index], output_margin=False, ntree_limit=3)
        labels = y[test_index]

        check_pred(preds, labels)
        check_pred(preds2, labels)
        check_pred(preds3, labels)
        check_pred(preds4, labels)


def test_feature_importances():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits

    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    xgb_model = xgb.XGBClassifier(seed=0).fit(X, y)

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
    xgb_model = xgb.XGBClassifier(seed=0).fit(X, y)
    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)

    # string columns, the feature order must be kept
    chars = list('abcdefghijklmnopqrstuvwxyz')
    X.columns = ["".join(random.sample(chars, 5)) for x in range(64)]

    xgb_model = xgb.XGBClassifier(seed=0).fit(X, y)
    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)


def test_boston_housing_regression():
    tm._skip_if_no_sklearn()
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import KFold

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf:
        xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])

        preds = xgb_model.predict(X[test_index])
        # test other params in XGBRegressor().fit
        preds2 = xgb_model.predict(X[test_index], output_margin=True, ntree_limit=3)
        preds3 = xgb_model.predict(X[test_index], output_margin=True, ntree_limit=0)
        preds4 = xgb_model.predict(X[test_index], output_margin=False, ntree_limit=3)
        labels = y[test_index]

        assert mean_squared_error(preds, labels) < 25
        assert mean_squared_error(preds2, labels) < 350
        assert mean_squared_error(preds3, labels) < 25
        assert mean_squared_error(preds4, labels) < 350


def test_parameter_tuning():
    tm._skip_if_no_sklearn()
    from sklearn.grid_search import GridSearchCV
    from sklearn.datasets import load_boston

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    xgb_model = xgb.XGBRegressor()
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                                   'n_estimators': [50, 100, 200]}, verbose=1)
    clf.fit(X, y)
    assert clf.best_score_ < 0.7
    assert clf.best_params_ == {'n_estimators': 100, 'max_depth': 4}


def test_regression_with_custom_objective():
    tm._skip_if_no_sklearn()
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import KFold

    def objective_ls(y_true, y_pred):
        grad = (y_pred - y_true)
        hess = np.ones(len(y_true))
        return grad, hess

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf:
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
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import KFold

    def logregobj(y_true, y_pred):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        grad = y_pred - y_true
        hess = y_pred * (1.0 - y_pred)
        return grad, hess

    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf:
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


def test_sklearn_api():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_iris
    from sklearn.cross_validation import train_test_split

    iris = load_iris()
    tr_d, te_d, tr_l, te_l = train_test_split(iris.data, iris.target, train_size=120)

    classifier = xgb.XGBClassifier(booster='gbtree', n_estimators=10)
    classifier.fit(tr_d, tr_l)

    preds = classifier.predict(te_d)
    labels = te_l
    err = sum([1 for p, l in zip(preds, labels) if p != l]) * 1.0 / len(te_l)
    assert err < 0.2


def test_sklearn_api_gblinear():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_iris
    from sklearn.cross_validation import train_test_split

    iris = load_iris()
    tr_d, te_d, tr_l, te_l = train_test_split(iris.data, iris.target, train_size=120)

    classifier = xgb.XGBClassifier(booster='gblinear', n_estimators=100)
    classifier.fit(tr_d, tr_l)

    preds = classifier.predict(te_d)
    labels = te_l
    err = sum([1 for p, l in zip(preds, labels) if p != l]) * 1.0 / len(te_l)
    assert err < 0.2


def test_sklearn_plotting():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_iris

    iris = load_iris()

    classifier = xgb.XGBClassifier()
    classifier.fit(iris.data, iris.target)

    import matplotlib
    matplotlib.use('Agg')

    from matplotlib.axes import Axes
    from graphviz import Digraph

    ax = xgb.plot_importance(classifier)
    assert isinstance(ax, Axes)
    assert ax.get_title() == 'Feature importance'
    assert ax.get_xlabel() == 'F score'
    assert ax.get_ylabel() == 'Features'
    assert len(ax.patches) == 4

    g = xgb.to_graphviz(classifier, num_trees=0)
    assert isinstance(g, Digraph)

    ax = xgb.plot_tree(classifier, num_trees=0)
    assert isinstance(ax, Axes)


def test_sklearn_nfolds_cv():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold

    digits = load_digits(3)
    X = digits['data']
    y = digits['target']
    dm = xgb.DMatrix(X, label=y)

    params = {
        'max_depth': 2,
        'eta': 1,
        'silent': 1,
        'objective':
        'multi:softprob',
        'num_class': 3
    }

    seed = 2016
    nfolds = 5
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

    cv1 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds, seed=seed)
    cv2 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds, folds=skf, seed=seed)
    cv3 = xgb.cv(params, dm, num_boost_round=10, nfold=nfolds, stratified=True, seed=seed)
    assert cv1.shape[0] == cv2.shape[0] and cv2.shape[0] == cv3.shape[0]
    assert cv2.iloc[-1, 0] == cv3.iloc[-1, 0]


def test_split_value_histograms():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits

    digits_2class = load_digits(2)

    X = digits_2class['data']
    y = digits_2class['target']

    dm = xgb.DMatrix(X, label=y)
    params = {'max_depth': 6, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}

    gbdt = xgb.train(params, dm, num_boost_round=10)
    assert gbdt.get_split_value_histogram("not_there", as_pandas=True).shape[0] == 0
    assert gbdt.get_split_value_histogram("not_there", as_pandas=False).shape[0] == 0
    assert gbdt.get_split_value_histogram("f28", bins=0).shape[0] == 1
    assert gbdt.get_split_value_histogram("f28", bins=1).shape[0] == 1
    assert gbdt.get_split_value_histogram("f28", bins=2).shape[0] == 2
    assert gbdt.get_split_value_histogram("f28", bins=5).shape[0] == 2
    assert gbdt.get_split_value_histogram("f28", bins=None).shape[0] == 2
