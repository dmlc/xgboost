################
Multiple Outputs
################

.. versionadded:: 1.6

Starting from version 1.6, XGBoost has experimental support for multi-output regression
and multi-label classification with Python package.  Multi-label classification usually
refers to targets that have multiple non-exclusive class labels.  For instance, a movie
can be simultaneously classified as both sci-fi and comedy.  For detailed explanation of
terminologies related to different multi-output models please refer to the
:doc:`scikit-learn user guide <sklearn:modules/multiclass>`.

Internally, XGBoost builds one model for each target similar to sklearn meta estimators,
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
