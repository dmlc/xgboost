################
Categorical Data
################

Starting from version 1.5, XGBoost has experimental support for categorical data available
for public testing.  At the moment, the support is implemented as one-hot encoding based
categorical tree splits.  For numerical data, the split condition is defined as
:math:`value < threshold`, while for categorical data the split is defined as :math:`value
== category` and ``category`` is a discrete value.  More advanced categorical split
strategy is planned for future releases and this tutorial details how to inform XGBoost
about the data type.  Also, the current support for training is limited to ``gpu_hist``
tree method.

************************************
Training with scikit-learn Interface
************************************

The easiest way to pass categorical data into XGBoost is using dataframe and the
``scikit-learn`` interface like :class:`XGBClassifier <xgboost.XGBClassifier>`.  For
preparing the data, users need to specify the data type of input predictor as
``category``.  For ``pandas/cudf Dataframe``, this can be achieved by

.. code:: python

  X["cat_feature"].astype("category")

for all columns that represent categorical features.  After which, users can tell XGBoost
to enable training with categorical data.  Assuming that you are using the
:class:`XGBClassifier <xgboost.XGBClassifier>` for classification problem, specify the
parameter ``enable_categorical``:

.. code:: python

  # Only gpu_hist is supported for categorical data as mentioned previously
  clf = xgb.XGBClassifier(
      tree_method="gpu_hist", enable_categorical=True, use_label_encoder=False
  )
  # X is the dataframe we created in previous snippet
  clf.fit(X, y)
  # Must use JSON for serialization, otherwise the information is lost
  clf.save_model("categorical-model.json")


Once training is finished, most of other features can utilize the model.  For instance one
can plot the model and calculate the global feature importance:


.. code:: python

  # Get a graph
  graph = xgb.to_graphviz(clf, num_trees=1)
  # Or get a matplotlib axis
  ax = xgb.plot_tree(clf, num_trees=1)
  # Get feature importances
  clf.feature_importances_


The ``scikit-learn`` interface from dask is similar to single node version.  The basic
idea is create dataframe with category feature type, and tell XGBoost to use ``gpu_hist``
with parameter ``enable_categorical``.  See :ref:`sphx_glr_python_examples_categorical.py`
for a worked example of using categorical data with ``scikit-learn`` interface.  A
comparison between using one-hot encoded data and XGBoost's categorical data support can
be found :ref:`sphx_glr_python_examples_cat_in_the_dat.py`.


**********************
Using native interface
**********************

The ``scikit-learn`` interface is user friendly, but lacks some features that are only
available in native interface.  For instance users cannot compute SHAP value directly or
use quantized :class:`DMatrix <xgboost.DMatrix>`.  Also native interface supports data
types other than dataframe, like ``numpy/cupy array``. To use the native interface with
categorical data, we need to pass the similar parameter to :class:`DMatrix
<xgboost.DMatrix>` and the :func:`train <xgboost.train>` function.  For dataframe input:

.. code:: python

  # X is a dataframe we created in previous snippet
  Xy = xgb.DMatrix(X, y, enable_categorical=True)
  booster = xgb.train({"tree_method": "gpu_hist"}, Xy)
  # Must use JSON for serialization, otherwise the information is lost
  booster.save_model("categorical-model.json")

SHAP value computation:

.. code:: python

  SHAP = booster.predict(Xy, pred_interactions=True)

  # categorical features are listed as "c"
  print(booster.feature_types)


For other types of input, like ``numpy array``, we can tell XGBoost about the feature
types by using the ``feature_types`` parameter in :class:`DMatrix <xgboost.DMatrix>`:

.. code:: python

  # "q" is numerical feature, while "c" is categorical feature
  ft = ["q", "c", "c"]
  X: np.ndarray = load_my_data()
  assert X.shape[1] == 3
  Xy = xgb.DMatrix(X, y, feature_types=ft, enable_categorical=True)

For numerical data, the feature type can be ``"q"`` or ``"float"``, while for categorical
feature it's specified as ``"c"``.  The Dask module in XGBoost has the same interface so
:class:`dask.Array <dask.Array>` can also be used as categorical data.


**********
Next Steps
**********

As of XGBoost 1.5, the feature is highly experimental and have limited features like CPU
training is not yet supported.  Please see `this issue
<https://github.com/dmlc/xgboost/issues/6503>`_ for progress.
