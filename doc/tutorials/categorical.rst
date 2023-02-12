################
Categorical Data
################

.. note::

   As of XGBoost 1.6, the feature is experimental and has limited features

Starting from version 1.5, XGBoost has experimental support for categorical data available
for public testing. For numerical data, the split condition is defined as :math:`value <
threshold`, while for categorical data the split is defined depending on whether
partitioning or onehot encoding is used. For partition-based splits, the splits are
specified as :math:`value \in categories`, where ``categories`` is the set of categories
in one feature.  If onehot encoding is used instead, then the split is defined as
:math:`value == category`. More advanced categorical split strategy is planned for future
releases and this tutorial details how to inform XGBoost about the data type.

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

  # Supported tree methods are `gpu_hist`, `approx`, and `hist`.
  clf = xgb.XGBClassifier(tree_method="gpu_hist", enable_categorical=True)
  # X is the dataframe we created in previous snippet
  clf.fit(X, y)
  # Must use JSON/UBJSON for serialization, otherwise the information is lost.
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
idea is create dataframe with category feature type, and tell XGBoost to use it by setting
the ``enable_categorical`` parameter.  See :ref:`sphx_glr_python_examples_categorical.py`
for a worked example of using categorical data with ``scikit-learn`` interface with
one-hot encoding.  A comparison between using one-hot encoded data and XGBoost's
categorical data support can be found :ref:`sphx_glr_python_examples_cat_in_the_dat.py`.


********************
Optimal Partitioning
********************

.. versionadded:: 1.6

Optimal partitioning is a technique for partitioning the categorical predictors for each
node split, the proof of optimality for numerical output was first introduced by `[1]
<#references>`__. The algorithm is used in decision trees `[2] <#references>`__, later
LightGBM `[3] <#references>`__ brought it to the context of gradient boosting trees and
now is also adopted in XGBoost as an optional feature for handling categorical
splits. More specifically, the proof by Fisher `[1] <#references>`__ states that, when
trying to partition a set of discrete values into groups based on the distances between a
measure of these values, one only needs to look at sorted partitions instead of
enumerating all possible permutations. In the context of decision trees, the discrete
values are categories, and the measure is the output leaf value.  Intuitively, we want to
group the categories that output similar leaf values. During split finding, we first sort
the gradient histogram to prepare the contiguous partitions then enumerate the splits
according to these sorted values. One of the related parameters for XGBoost is
``max_cat_to_onehot``, which controls whether one-hot encoding or partitioning should be
used for each feature, see :ref:`cat-param` for details.


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
  booster = xgb.train({"tree_method": "hist", "max_cat_to_onehot": 5}, Xy)
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
:class:`dask.Array <dask.Array>` can also be used for categorical data.

*************
Miscellaneous
*************

By default, XGBoost assumes input categories are integers starting from 0 till the number
of categories :math:`[0, n\_categories)`. However, user might provide inputs with invalid
values due to mistakes or missing values in training dataset. It can be negative value,
integer values that can not be accurately represented by 32-bit floating point, or values
that are larger than actual number of unique categories.  During training this is
validated but for prediction it's treated as the same as not-chosen category for
performance reasons.


**********
References
**********

[1] Walter D. Fisher. "`On Grouping for Maximum Homogeneity`_". Journal of the American Statistical Association. Vol. 53, No. 284 (Dec., 1958), pp. 789-798.

[2] Trevor Hastie, Robert Tibshirani, Jerome Friedman. "`The Elements of Statistical Learning`_". Springer Series in Statistics Springer New York Inc. (2001).

[3] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "`LightGBM\: A Highly Efficient Gradient Boosting Decision Tree`_." Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.


.. _On Grouping for Maximum Homogeneity: https://www.tandfonline.com/doi/abs/10.1080/01621459.1958.10501479

.. _The Elements of Statistical Learning: https://link.springer.com/book/10.1007/978-0-387-84858-7

.. _LightGBM\: A Highly Efficient Gradient Boosting Decision Tree: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf
