################
Multiple Outputs
################

.. versionadded:: 1.6

Starting from version 1.6, XGBoost has experimental support for multi-output regression
and multi-label classification.  For the terminologies please refer to the `scikit-learn
user guide <https://scikit-learn.org/stable/modules/multiclass.html>`_.

Internally, XGBoost builds one model for each target similar to sklearn meta estimators,
with the added benefit of reusing data and custom objective support.  For a worked example
of regression, see :ref:`sphx_glr_python_examples_multioutput_regression.py`. For
multi-label classification, the binary relevance strategy is used.  Since classes are not
mutually exclusive so XGBoost will train one binary classifier for each target. Input
``y`` should be of shape ``(n_samples, n_classes)`` with each column has value 0 or 1 to
specify whether the sample is labeled as positive.

.. code-block:: python

    from sklearn.datasets import make_multilabel_classification
    import numpy as np

    X, y = make_multilabel_classification(
        n_samples=32, n_classes=5, n_labels=3, random_state=0
    )
    clf = xgb.XGBClassifier(tree_method="hist")
    clf.fit(X, y)
    np.testing.assert_allclose(clf.predict(X), y)


The feature is still under development and might contain unknown issues.
