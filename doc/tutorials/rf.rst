#############################
Random Forests(TM) in XGBoost
#############################

XGBoost is normally used to train gradient-boosted decision trees and other gradient
boosted models. Random Forests use the same model representation and inference, as
gradient-boosted decision trees, but a different training algorithm.  One can use XGBoost
to train a standalone random forest or use random forest as a base model for gradient
boosting.  Here we focus on training standalone random forest.

We have native APIs for training random forests since the early days, and a new
Scikit-Learn wrapper after 0.82 (not included in 0.82).  Please note that the new
Scikit-Learn wrapper is still **experimental**, which means we might change the interface
whenever needed.

*****************************************
Standalone Random Forest With XGBoost API
*****************************************

The following parameters must be set to enable random forest training.

* ``booster`` should be set to ``gbtree``, as we are training forests. Note that as this
  is the default, this parameter needn't be set explicitly.
* ``subsample`` must be set to a value less than 1 to enable random selection of training
  cases (rows).
* One of ``colsample_by*`` parameters must be set to a value less than 1 to enable random
  selection of columns. Normally, ``colsample_bynode`` would be set to a value less than 1
  to randomly sample columns at each tree split.
* ``num_parallel_tree`` should be set to the size of the forest being trained.
* ``num_boost_round`` should be set to 1 to prevent XGBoost from boosting multiple random
  forests.  Note that this is a keyword argument to ``train()``, and is not part of the
  parameter dictionary.
* ``eta`` (alias: ``learning_rate``) must be set to 1 when training random forest
  regression.
* ``random_state`` can be used to seed the random number generator.


Other parameters should be set in a similar way they are set for gradient boosting. For
instance, ``objective`` will typically be ``reg:squarederror`` for regression and
``binary:logistic`` for classification, ``lambda`` should be set according to a desired
regularization weight, etc.

If both ``num_parallel_tree`` and ``num_boost_round`` are greater than 1, training will
use a combination of random forest and gradient boosting strategy. It will perform
``num_boost_round`` rounds, boosting a random forest of ``num_parallel_tree`` trees at
each round. If early stopping is not enabled, the final model will consist of
``num_parallel_tree`` * ``num_boost_round`` trees.

Here is a sample parameter dictionary for training a random forest on a GPU using
xgboost::

  params = {
    'colsample_bynode': 0.8,
    'learning_rate': 1,
    'max_depth': 5,
    'num_parallel_tree': 100,
    'objective': 'binary:logistic',
    'subsample': 0.8,
    'tree_method': 'gpu_hist'
  }

A random forest model can then be trained as follows::

  bst = train(params, dmatrix, num_boost_round=1)


***************************************************
Standalone Random Forest With Scikit-Learn-Like API
***************************************************

``XGBRFClassifier`` and ``XGBRFRegressor`` are SKL-like classes that provide random forest
functionality. They are basically versions of ``XGBClassifier`` and ``XGBRegressor`` that
train random forest instead of gradient boosting, and have default values and meaning of
some of the parameters adjusted accordingly. In particular:

* ``n_estimators`` specifies the size of the forest to be trained; it is converted to
  ``num_parallel_tree``, instead of the number of boosting rounds
* ``learning_rate`` is set to 1 by default
* ``colsample_bynode`` and ``subsample`` are set to 0.8 by default
* ``booster`` is always ``gbtree``

For a simple example, you can train a random forest regressor with::

    from sklearn.model_selection import KFold

    # Your code ...

    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBRFRegressor(random_state=42).fit(
	X[train_index], y[train_index])

Note that these classes have a smaller selection of parameters compared to using
``train()``. In particular, it is impossible to combine random forests with gradient
boosting using this API.


*******
Caveats
*******

* XGBoost uses 2nd order approximation to the objective function. This can lead to results
  that differ from a random forest implementation that uses the exact value of the
  objective function.
* XGBoost does not perform replacement when subsampling training cases. Each training case
  can occur in a subsampled set either 0 or 1 time.
