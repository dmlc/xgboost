################
Multiple Outputs
################

**Contents**

.. contents::
  :backlinks: none
  :local:


.. versionadded:: 1.6

Starting from version 1.6, XGBoost has experimental support for multi-output regression
and multi-label classification with Python package.  Multi-label classification usually
refers to targets that have multiple non-exclusive class labels.  For instance, a movie
can be simultaneously classified as both sci-fi and comedy.  For detailed explanation of
terminologies related to different multi-output models please refer to the
:doc:`scikit-learn user guide <sklearn:modules/multiclass>`.

.. note:: As of XGBoost 3.4.0, the feature is experimental.

   The ``hist`` tree method is feature complete. In addition, the Python interface along with the Dask interface are supported. See the end of the doc for a brief history.

**********************************
Training with One-Model-Per-Target
**********************************

By default, XGBoost builds one model for each target similar to sklearn meta estimators,
with the added benefit of reusing data and other integrated features like SHAP.  For a
worked example of regression, see
:ref:`sphx_glr_python_examples_multioutput_regression.py`. For multi-label classification,
the binary relevance strategy is used.  Input ``y`` should be of shape ``(n_samples,
n_classes)`` with each column having a value of 0 or 1 to specify whether the sample is
labeled as positive for respective class. Given a sample with 3 output classes and 2
labels, the corresponding `y` should be encoded as ``[1, 0, 1]`` with the second class
labeled as negative and the rest labeled as positive. At the moment XGBoost supports only
dense matrix for labels.

.. code-block:: python

    from sklearn.datasets import make_multilabel_classification
    import numpy as np

    X, y = make_multilabel_classification(
        n_samples=32, n_classes=5, n_labels=3, random_state=0
    )
    clf = xgb.XGBClassifier(tree_method="hist")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), y)


The feature is still under development with limited support from objectives and metrics.

*************************
Training with Vector Leaf
*************************

.. versionadded:: 2.0.0

XGBoost can optionally build multi-output trees with the size of leaf equals to the number
of targets when the tree method `hist` is used. The behavior can be controlled by the
``multi_strategy`` training parameter, which can take the value `one_output_per_tree` (the
default) for building one model per-target or `multi_output_tree` for building
multi-output trees.

.. code-block:: python

  clf = xgb.XGBClassifier(tree_method="hist", multi_strategy="multi_output_tree")

See :ref:`sphx_glr_python_examples_multioutput_regression.py` for a worked example with
regression.


*******************************
Random Projection for Splitting
*******************************

.. versionadded:: 3.5.0

For vector-leaf models with many outputs, ``multi_projection_ratio`` reduces the number of
targets needed for split search using random projection, following the split/value
gradient separation in `Sketch Boost` `[1] <#references>`__.

Split search uses projected gradients and Hessians. Leaf values are fitted using the
original gradients and Hessians for every output. The following snippet uses one quarter
of the target dimensions for split search. See :doc:`/parameter` for valid ratios and
restrictions.

.. code-block:: python

    reg = xgb.XGBRegressor(
        tree_method="hist",
        multi_strategy="multi_output_tree",
        multi_projection_ratio=0.25,
    )
    reg.fit(X, y)


Reducing the number of targets can change the effect of regularization parameters. The
effective ``gamma`` is scaled by the retained dimension fraction to approximate the change
in split-gain scale. The ``reg_lambda`` retains its L2 regularization meaning and requires
no rescaling under random projection. Other regularization parameters like L1
(``reg_alpha``) are not invariant to the random projection.

*************
Brief History
*************
Some milestones of multi-output support are recorded below:

- XGBoost introduced basic concepts for multi-output in v1.6.0.
- v2.0.0 introduced a prototype for training CPU-based vector-leaf models, which established the vector-leaf model format.
- v3.2.0 saw a major upgrade to the prototype for CUDA implementation, added reduced gradient, vector intercept, along with external memory.
- v3.4.0 completes the ``hist`` tree method implementation, for both CPU, GPU, distributed training (Dask), and categorical features, along with various metrics, objectives, and model visualization.
- v3.5.0 replaces custom gradient-reduction callbacks with native random projection.

**********
References
**********

[1] Leonid Iosipoi, Anton Vakhrushev. "`Fast Gradient Boosted Decision Tree for Multioutput Problems`_". NeurIPS 2022, pp 25422 - 25435.

.. _Fast Gradient Boosted Decision Tree for Multioutput Problems: https://proceedings.neurips.cc/paper_files/paper/2022/file/a36c3dbe676fa8445715a31a90c66ab3-Paper-Conference.pdf
