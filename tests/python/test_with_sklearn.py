import numpy as np
import xgboost as xgb
import testing as tm
import tempfile
import os
import shutil
from nose.tools import raises

rng = np.random.RandomState(1994)


class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp()"""
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)


def test_binary_classification():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.1


def test_multiclass_classification():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_iris
    from sklearn.model_selection import KFold

    def check_pred(preds, labels, output_margin):
        if output_margin:
            err = sum(1 for i in range(len(preds))
                      if preds[i].argmax() != labels[i]) / float(len(preds))
        else:
            err = sum(1 for i in range(len(preds))
                      if preds[i] != labels[i]) / float(len(preds))
        assert err < 0.4

    iris = load_iris()
    y = iris['target']
    X = iris['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
        preds = xgb_model.predict(X[test_index])
        # test other params in XGBClassifier().fit
        preds2 = xgb_model.predict(X[test_index], output_margin=True, ntree_limit=3)
        preds3 = xgb_model.predict(X[test_index], output_margin=True, ntree_limit=0)
        preds4 = xgb_model.predict(X[test_index], output_margin=False, ntree_limit=3)
        labels = y[test_index]

        check_pred(preds, labels, output_margin=False)
        check_pred(preds2, labels, output_margin=True)
        check_pred(preds3, labels, output_margin=True)
        check_pred(preds4, labels, output_margin=False)


def test_ranking():
    tm._skip_if_no_sklearn()
    # generate random data
    x_train = np.random.rand(1000, 10)
    y_train = np.random.randint(5, size=1000)
    train_group = np.repeat(50, 20)
    x_valid = np.random.rand(200, 10)
    y_valid = np.random.randint(5, size=200)
    valid_group = np.repeat(50, 4)
    x_test = np.random.rand(100, 10)

    params = {'objective': 'rank:pairwise', 'learning_rate': 0.1,
              'gamma': 1.0, 'min_child_weight': 0.1,
              'max_depth': 6, 'n_estimators': 4}
    model = xgb.sklearn.XGBRanker(**params)
    model.fit(x_train, y_train, train_group,
              eval_set=[(x_valid, y_valid)], eval_group=[valid_group])
    pred = model.predict(x_test)

    train_data = xgb.DMatrix(x_train, y_train)
    valid_data = xgb.DMatrix(x_valid, y_valid)
    test_data = xgb.DMatrix(x_test)
    train_data.set_group(train_group)
    valid_data.set_group(valid_group)

    params_orig = {'objective': 'rank:pairwise', 'eta': 0.1, 'gamma': 1.0,
                   'min_child_weight': 0.1, 'max_depth': 6}
    xgb_model_orig = xgb.train(params_orig, train_data, num_boost_round=4,
                               evals=[(valid_data, 'validation')])
    pred_orig = xgb_model_orig.predict(test_data)

    np.testing.assert_almost_equal(pred, pred_orig)


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

    xgb_model = xgb.XGBClassifier(seed=0).fit(X, y)
    np.testing.assert_almost_equal(xgb_model.feature_importances_, exp)


def test_boston_housing_regression():
    tm._skip_if_no_sklearn()
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.model_selection import KFold

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
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
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import load_boston

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    xgb_model = xgb.XGBRegressor()
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                                   'n_estimators': [50, 100, 200]},
                       cv=3, verbose=1, iid=True)
    clf.fit(X, y)
    assert clf.best_score_ < 0.7
    assert clf.best_params_ == {'n_estimators': 100, 'max_depth': 4}


def test_regression_with_custom_objective():
    tm._skip_if_no_sklearn()
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.model_selection import KFold

    def objective_ls(y_true, y_pred):
        grad = (y_pred - y_true)
        hess = np.ones(len(y_true))
        return grad, hess

    boston = load_boston()
    y = boston['target']
    X = boston['data']
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
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    def logregobj(y_true, y_pred):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        grad = y_pred - y_true
        hess = y_pred * (1.0 - y_pred)
        return grad, hess

    digits = load_digits(2)
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


def test_sklearn_api():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    tr_d, te_d, tr_l, te_l = train_test_split(iris.data, iris.target,
                                              train_size=120, test_size=0.2)

    classifier = xgb.XGBClassifier(booster='gbtree', n_estimators=10)
    classifier.fit(tr_d, tr_l)

    preds = classifier.predict(te_d)
    labels = te_l
    err = sum([1 for p, l in zip(preds, labels) if p != l]) * 1.0 / len(te_l)
    assert err < 0.2


def test_sklearn_api_gblinear():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    tr_d, te_d, tr_l, te_l = train_test_split(iris.data, iris.target, train_size=120)

    classifier = xgb.XGBClassifier(booster='gblinear', n_estimators=100)
    classifier.fit(tr_d, tr_l)

    preds = classifier.predict(te_d)
    labels = te_l
    err = sum([1 for p, l in zip(preds, labels) if p != l]) * 1.0 / len(te_l)
    assert err < 0.5


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


def test_sklearn_random_state():
    tm._skip_if_no_sklearn()

    clf = xgb.XGBClassifier(random_state=402)
    assert clf.get_xgb_params()['seed'] == 402

    clf = xgb.XGBClassifier(seed=401)
    assert clf.get_xgb_params()['seed'] == 401


def test_sklearn_n_jobs():
    tm._skip_if_no_sklearn()

    clf = xgb.XGBClassifier(n_jobs=1)
    assert clf.get_xgb_params()['nthread'] == 1

    clf = xgb.XGBClassifier(nthread=2)
    assert clf.get_xgb_params()['nthread'] == 2


def test_kwargs():
    tm._skip_if_no_sklearn()

    params = {'updater': 'grow_gpu', 'subsample': .5, 'n_jobs': -1}
    clf = xgb.XGBClassifier(n_estimators=1000, **params)
    assert clf.get_params()['updater'] == 'grow_gpu'
    assert clf.get_params()['subsample'] == .5
    assert clf.get_params()['n_estimators'] == 1000


def test_kwargs_grid_search():
    tm._skip_if_no_sklearn()
    from sklearn.model_selection import GridSearchCV
    from sklearn import datasets

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


@raises(TypeError)
def test_kwargs_error():
    tm._skip_if_no_sklearn()

    params = {'updater': 'grow_gpu', 'subsample': .5, 'n_jobs': -1}
    clf = xgb.XGBClassifier(n_jobs=1000, **params)
    assert isinstance(clf, xgb.XGBClassifier)


def test_sklearn_clone():
    tm._skip_if_no_sklearn()
    from sklearn.base import clone

    clf = xgb.XGBClassifier(n_jobs=2, nthread=3)
    clf.n_jobs = -1
    clone(clf)


def test_validation_weights_xgbmodel():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import make_hastie_10_2

    # prepare training and test data
    X, y = make_hastie_10_2(n_samples=2000, random_state=42)
    labels, y = np.unique(y, return_inverse=True)
    X_train, X_test = X[:1600], X[1600:]
    y_train, y_test = y[:1600], y[1600:]

    # instantiate model
    param_dist = {'objective': 'binary:logistic', 'n_estimators': 2,
                  'random_state': 123}
    clf = xgb.sklearn.XGBModel(**param_dist)

    # train it using instance weights only in the training set
    weights_train = np.random.choice([1, 2], len(X_train))
    clf.fit(X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_metric='logloss',
            verbose=False)

    # evaluate logloss metric on test set *without* using weights
    evals_result_without_weights = clf.evals_result()
    logloss_without_weights = evals_result_without_weights["validation_0"]["logloss"]

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

    # check that the logloss in the test set is actually different when using weights
    # than when not using them
    assert all((logloss_with_weights[i] != logloss_without_weights[i] for i in [0, 1]))


def test_validation_weights_xgbclassifier():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import make_hastie_10_2

    # prepare training and test data
    X, y = make_hastie_10_2(n_samples=2000, random_state=42)
    labels, y = np.unique(y, return_inverse=True)
    X_train, X_test = X[:1600], X[1600:]
    y_train, y_test = y[:1600], y[1600:]

    # instantiate model
    param_dist = {'objective': 'binary:logistic', 'n_estimators': 2,
                  'random_state': 123}
    clf = xgb.sklearn.XGBClassifier(**param_dist)

    # train it using instance weights only in the training set
    weights_train = np.random.choice([1, 2], len(X_train))
    clf.fit(X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_metric='logloss',
            verbose=False)

    # evaluate logloss metric on test set *without* using weights
    evals_result_without_weights = clf.evals_result()
    logloss_without_weights = evals_result_without_weights["validation_0"]["logloss"]

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

    # check that the logloss in the test set is actually different when using weights
    # than when not using them
    assert all((logloss_with_weights[i] != logloss_without_weights[i] for i in [0, 1]))


def test_save_load_model():
    tm._skip_if_no_sklearn()
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    with TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, 'digits.model')
        for train_index, test_index in kf.split(X, y):
            xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
            xgb_model.save_model(model_path)
            xgb_model = xgb.XGBModel()
            xgb_model.load_model(model_path)
            preds = xgb_model.predict(X[test_index])
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1
