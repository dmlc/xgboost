##########################################
Using the Scikit-Learn Estimator Interface
##########################################

**Contents**

.. contents::
  :backlinks: none
  :local:

********
Overview
********

In addition to the native interface, XGBoost features a sklearn estimator interface that
conforms to `sklearn estimator guideline
<https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`__. It
supports regression, classification, and learning to rank. Survival training for the
sklearn estimator interface is still working in progress.

You can find some some quick start examples at
:ref:`sphx_glr_python_examples_sklearn_examples.py`. The main advantage of using sklearn
interface is that it works with most of the utilites provided by sklearn like
:py:func:`sklearn.model_selection.cross_validate`. Also, many other libraries recognize
the sklearn estimator interface thanks to its popularity.

With the sklearn estimator interface, we can train a classification model with only a
couple lines of Python code. Here's an example for training a classification model:

.. code-block:: python

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    import xgboost as xgb

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=94)

    # Use "hist" for constructing the trees, with early stopping enabled.
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    # Fit the model, test sets are used for early stopping.
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    # Save model into JSON format.
    clf.save_model("clf.json")


The ``tree_method`` parameter specifies the method to use for constructing the trees, and
the early_stopping_rounds parameter enables early stopping. Early stopping can help
prevent overfitting and save time during training.

**************
Early Stopping
**************

As demonstrated in the previous example, early stopping can be enabled by the parameter
``early_stopping_rounds``. Alternatively, there's a callback function that can be used
:py:class:`xgboost.callback.EarlyStopping` to specify more details about the behavior of
early stopping, including whether XGBoost should return the best model instead of the full
stack of trees:

.. code-block:: python

    early_stop = xgb.callback.EarlyStopping(
        rounds=2, metric_name='logloss', data_name='Validation_0', save_best=True
    )
    clf = xgb.XGBClassifier(tree_method="hist", callbacks=[early_stop])
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

At present, XGBoost doesn't implement data spliting logic within the estimator and relies
on the ``eval_set`` parameter of the :py:meth:`xgboost.XGBModel.fit` method. If you want
to use early stopping to prevent overfitting, you'll need to manually split your data into
training and testing sets using the :py:func:`sklearn.model_selection.train_test_split`
function from the `sklearn` library. Some other machine learning algorithms, like those in
`sklearn`, include early stopping as part of the estimator and may work with cross
validation. However, using early stopping during cross validation may not be a perfect
approach because it changes the model's number of trees for each validation fold, leading
to different model. A better approach is to retrain the model after cross validation using
the best hyperparameters along with early stopping. If you want to experiment with idea of
using cross validation with early stopping, here is a snippet to begin with:

.. code-block:: python

    from sklearn.base import clone
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold, cross_validate

    import xgboost as xgb

    X, y = load_breast_cancer(return_X_y=True)


    def fit_and_score(estimator, X_train, X_test, y_train, y_test):
        """Fit the estimator on the train set and score it on both sets"""
        estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        train_score = estimator.score(X_train, y_train)
        test_score = estimator.score(X_test, y_test)

        return estimator, train_score, test_score


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)

    resutls = {}

    for train, test in cv.split(X, y):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        est, train_score, test_score = fit_and_score(
            clone(clf), X_train, X_test, y_train, y_test
        )
        resutls[est] = (train_score, test_score)


***********************************
Obtaining the native booster object
***********************************

The sklearn estimator interface primarily facilitates training and doesn't implement all
features available in XGBoost. For instance, in order to have cached predictions,
:py:class:`xgboost.DMatrix` needs to be used with :py:meth:`xgboost.Booster.predict`. One
can obtain the booster object from the sklearn interface using
:py:meth:`xgboost.XGBModel.get_booster`:

.. code-block:: python

   booster = clf.get_booster()
   print(booster.num_boosted_rounds())


**********
Prediction
**********

When early stopping is enabled, prediction functions including the
:py:meth:`xgboost.XGBModel.predict`, :py:meth:`xgboost.XGBModel.score`, and
:py:meth:`xgboost.XGBModel.apply` methods will use the best model automatically. Meaning
the :py:attr:`xgboost.XGBModel.best_iteration` is used to specify the range of trees used
in prediction.

To have cached results for incremental prediction, please use the
:py:meth:`xgboost.Booster.predict` method instead.


**************************
Number of parallel threads
**************************

When working with XGBoost and other sklearn tools, you can specify how many threads you
want to use by using the ``n_jobs`` parameter. By default, XGBoost uses all the available
threads on your computer, which can lead to some interesting consequences when combined
with other sklearn functions like :py:func:`sklearn.model_selection.cross_validate`. If
both XGBoost and sklearn are set to use all threads, your computer may start to slow down
significantly due to something called "thread thrashing". To avoid this, you can simply
set the ``n_jobs`` parameter for XGBoost to `None` (which uses all threads) and the
``n_jobs`` parameter for sklearn to `1`. This way, both programs will be able to work
together smoothly without causing any unnecessary computer strain.
