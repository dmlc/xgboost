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


*************************************
Using Reduced Gradient (Sketch Boost)
*************************************

.. versionadded:: 3.2.0

.. note::

   This is experimental. It is documented here for early testers to provide feedback. Related
   interface might change without notice.

When the number of targets is large, training a gradient boosting tree model using the
full gradient matrix becomes challenging. The training procedure may run out of memory for
storing the histogram, or run extremely slowly due to the amount of computation needed. As
an optimization, XGBoost implements an interface for using two types of gradients based on
the concepts from `Sketch Boost` `[1] <#references>`__.

The key insight is that we can use different gradients for two distinct purposes:

- **Split gradient**: A reduced-dimension gradient used to determine the tree structure.
- **Value gradient**: The full gradient used to calculate the final leaf values for
  accurate predictions.

This separation allows the expensive histogram building and split finding to operate on a
smaller gradient matrix, while still producing valid predictions using the full loss
function for leaf values. The `Sketch Boost` paper proposes using dimensionality reduction
on the gradient matrix. In practice, one can also define a different but related loss with
a small gradient matrix for finding the tree structure.

To access this feature, create a custom objective that inherits from ``TreeObjective`` and
implement the ``split_grad`` method.

.. code-block:: python

    from xgboost.objective import TreeObjective
    from cuml.decomposition import TruncatedSVD

    import cupy as cp

    class LsObj(TreeObjective):
        def __call__(self, iteration: int, y_pred, dtrain):
            """Least squared error."""
            y_true = dtrain.get_label()
            grad = y_pred - y_true
            hess = cp.ones(grad.shape)
            return cp.array(grad), cp.array(hess)

        def split_grad(self, iteration: int, grad, hess):
            svd_params = {"algorithm": "jacobi", "n_components": 2, "n_iter": 8}
            svd = TruncatedSVD(output_type="cupy", **svd_params)
            svd.fit(grad)
            grad = svd.transform(grad)
            hess = svd.transform(hess)
            hess = cp.clip(hess, 0.01, None)

            return grad, hess

See :ref:`sphx_glr_python_examples_multioutput_reduced_gradient.py` for a complete worked
example. The feature supports only the ``multi_strategy=multi_output_tree``.

********************************************
Exact Hessian for multi-class (experimental)
********************************************

.. versionadded:: 3.2.0

XGBoost approximates the multi-class second-order statistics with a **diagonal** matrix: each
class gets its own scalar Hessian and the coupling between classes is discarded. The true
multinomial Hessian is dense,

.. math::

   g = p - y, \qquad H = \mathrm{diag}(p) - p p^{	op}

and its off-diagonal entries :math:`-p_i p_j` are exactly that discarded coupling. Setting
``multi_hessian=exact`` uses the dense matrix and solves one joint Newton system per leaf.

Because softmax is invariant to adding a constant to every logit, :math:`H` is singular
(:math:`H \mathbf{1} = 0`). XGBoost removes that direction by pinning the last class to logit
zero, solving in :math:`K-1` free coordinates, and writing the mean-centred representative of
the resulting step back to the :math:`K` leaf outputs. The L2 penalty is applied to that
centred leaf, :math:`R = \lambda (I - \mathbf{1}\mathbf{1}^{	op}/K)`, which makes the fitted
model independent of which class is used as the reference.

**Cost.** Histogram storage per bin grows from :math:`O(K)` to :math:`O(K^2)` and each leaf
solve costs :math:`O(K^3)`. End-to-end training is several times slower than ``diagonal``,
and the factor grows with ``num_class``; ``research/benchmark_k_scaling.py`` measures it.

**Benefit.** This is the open question, and it is deliberately not answered by assertion
here. ``research/benchmark_v2.py`` compares the two modes under a fixed protocol -- identical
grid, seeds and splits for both, configuration selection and early stopping on a validation
split, symmetric validation-only targets, and the test split evaluated once after selection.
It reports rounds-to-target, wall-clock-to-target, equal-wall-clock quality and held-out
quality, so the trade-off can be measured on your own data rather than inherited from someone
else's. Treat ``multi_hessian=exact`` as an opt-in experimental capability, not as a better
default.

**Requirements.** ``multi:softprob`` or ``multi:softmax``, ``device=cpu``,
``tree_method=hist``, ``multi_strategy=multi_output_tree``. Unsupported combinations are
rejected with an actionable error rather than silently falling back to the diagonal
curvature. ``min_child_weight`` is compared against each mode's own curvature and therefore
does not gate identically in the two modes; see :doc:`/parameter`.

``multi_hessian`` is a training parameter. It is not stored in the model, and a model trained
with it predicts identically to any other model.

*************
Brief History
*************
Some milestones of multi-output support are recorded below:

- XGBoost introduced basic concepts for multi-output in v1.6.0.
- v2.0.0 introduced a prototype for training CPU-based vector-leaf models, which established the vector-leaf model format.
- v3.2.0 saw a major upgrade to the prototype for CUDA implementation, added reduced gradient, vector intercept, along with external memory.
- v3.4.0 completes the ``hist`` tree method implementation, for both CPU, GPU, distributed training (Dask), and categorical features, along with various metrics, objectives, and model visualization.

**********
References
**********

[1] Leonid Iosipoi, Anton Vakhrushev. "`Fast Gradient Boosted Decision Tree for Multioutput Problems`_". NeurIPS 2022, pp 25422 - 25435.

.. _Fast Gradient Boosted Decision Tree for Multioutput Problems: https://proceedings.neurips.cc/paper_files/paper/2022/file/a36c3dbe676fa8445715a31a90c66ab3-Paper-Conference.pdf
