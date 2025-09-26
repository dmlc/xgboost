################
Categorical Data
################

Since version 1.5, XGBoost has support for categorical data.  For numerical data, the
split condition is defined as :math:`value < threshold`, while for categorical data the
split is defined depending on whether partitioning or onehot encoding is used. For
partition-based splits, the splits are specified as :math:`value \in categories`, where
``categories`` is the set of categories in one feature.  If onehot encoding is used
instead, then the split is defined as :math:`value == category`. More advanced categorical
split strategy is planned for future releases and this tutorial details how to inform
XGBoost about the data type.


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

  # Supported tree methods are `approx` and `hist`.
  clf = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, device="cuda")
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

.. versionadded:: 3.0

   Support for the R package using ``factor``.

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
available in native interface.  For instance users cannot compute SHAP value directly.
Also native interface supports more data types. To use the native interface with
categorical data, we need to pass the similar parameter to :class:`~xgboost.DMatrix` or
:py:class:`~xgboost.QuantileDMatrix` and the :func:`train <xgboost.train>` function.  For
dataframe input:

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
:class:`dask.Array <dask.Array>` can also be used for categorical data. Lastly, the
sklearn interface :py:class:`~xgboost.XGBRegressor` has the same parameter.

.. _cat-recode:

********************************
Auto-recoding (Data Consistency)
********************************

.. versionchanged:: 3.1

  Starting with XGBoost 3.1, the *Python* interface can perform automatic re-coding for
  new inputs.

XGBoost accepts parameters to indicate which feature is considered categorical, either
through the ``dtypes`` of a dataframe or through the ``feature_types`` parameter. However,
except for the Python interface, XGBoost doesn't store the information about how
categories are encoded in the first place. For instance, given an encoding schema that
maps music genres to integer codes:

.. code-block:: python

  {"acoustic": 0, "indie": 1, "blues": 2, "country": 3}

Aside from the Python interface (R/Java/C, etc), XGBoost doesn't know this mapping from
the input and hence cannot store it in the model. The mapping usually happens in the
users' data engineering pipeline. To ensure the correct result from XGBoost, users need to
keep the pipeline for transforming data consistent across training and testing data.

Starting with 3.1, the *Python* interface can remember the encoding and perform recoding
during inference and training continuation when the input is a dataframe (`pandas`,
`cuDF`, `polars`, `pyarrow`, `modin`). The feature support focuses on basic usage. It has
some restrictions on the types of inputs that can be accepted. First, category names
must have one of the following types:

- string
- integer, from 8-bit to 64-bit, both signed and unsigned are supported.
- 32-bit or 64-bit floating point

Other category types are not supported. Second, the input types must be strictly
consistent. For example, XGBoost will raise an error if the categorical columns in the
training set are unsigned integers whereas the test dataset has signed integer columns. If
you have categories that are not one of the supported types, you need to perform the
re-coding using a pre-processing data transformer like the
:py:class:`sklearn.preprocessing.OrdinalEncoder`. See
:ref:`sphx_glr_python_examples_cat_pipeline.py` for a worked example using an ordinal
encoder. To clarify, the type here refers to the type of the name of categories (called
``Index`` in pandas):

.. code-block:: python

  # string type
  {"acoustic": 0, "indie": 1, "blues": 2, "country": 3}
  # integer type
  {-1: 0, 1: 1, 3: 2, 7: 3}
  # depending on the dataframe implementation, it can be signed or unsigned.
  {5: 0, 1: 1, 3: 2, 7: 3}
  # floating point type, both 32-bit and 64-bit are supported.
  {-1.0: 0, 1.0: 1, 3.0: 2, 7.0: 3}

Internally, XGBoost attempts to extract the categories from the dataframe inputs. For
inference (predict), the re-coding happens on the fly and there's no data copy (baring
some internal transformations performed by the dataframe itself). For training
continuation however, re-coding requires some extra steps if you are using the native
interface. The sklearn interface and the Dask interface can handle training continuation
automatically. Last, please note that using the re-coder with the native interface is
still experimental. It's ready for testing, but we want to observe the feature usage for a
period of time and might make some breaking changes if needed. The following is a snippet
of using the native interface:

.. code-block:: python

  import pandas as pd

  X = pd.DataFrame()
  Xy = xgboost.QuantileDMatrix(X, y, enable_categorical=True)
  booster = xgboost.train({}, Xy)

  # XGBoost can handle re-coding for inference without user intervention
  X_new = pd.DataFrame()
  booster.inplace_predict(X_new)

  # Get categories saved in the model for training continuation
  categories = booster.get_categories()
  # Use saved categories as a reference for re-coding.
  # Training continuation requires a re-coded DMatrix, pass the categories as feature_types
  Xy_new = xgboost.QuantileDMatrix(
    X_new, y_new, feature_types=categories, enable_categorical=True, ref=Xy
  )
  booster_1 = xgboost.train({}, Xy_new, xgb_model=booster)


No extra step is required for using the scikit-learn interface as long as the inputs are
dataframes. During training continuation, XGBoost will either extract the categories from
the previous model or use the categories from the new training dataset if the input model
doesn't have the information.

For R, the auto-recoding is not yet supported as of 3.1. To provide an example:

.. code-block:: R

    > f0 = factor(c("a", "b", "c"))
    > as.numeric(f0)
    [1] 1 2 3
    > f0
    [1] a b c
    Levels: a b c

In the above snippet, we have the mapping: ``a -> 1, b -> 2, c -> 3``. Assuming the above
is the training data, and the next snippet is the test data:

.. code-block:: R

    > f1 = factor(c("a", "c"))
    > as.numeric(f1)
    [1] 1 2
    > f1
    [1] a c
    Levels: a c


Now, we have ``a -> 1, c -> 2`` because ``b`` is missing, and the R factor encodes the data
differently, resulting in invalid test-time encoding. XGBoost cannot remember the original
encoding for the R package. You will have to encode the data explicitly during inference:

.. code-block:: R

    > f1 = factor(c("a", "c"), levels = c("a", "b", "c"))
    > f1
    [1] a c
    Levels: a b c
    > as.numeric(f1)
      [1] 1 3


*************
Miscellaneous
*************

By default, XGBoost assumes input category codes are integers starting from 0 till the
number of categories :math:`[0, n\_categories)`. However, user might provide inputs with
invalid values due to mistakes or missing values in training dataset. It can be negative
value, integer values that can not be accurately represented by 32-bit floating point, or
values that are larger than actual number of unique categories.  During training this is
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
