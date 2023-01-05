######################################
Custom Objective and Evaluation Metric
######################################

**Contents**

.. contents::
  :backlinks: none
  :local:

********
Overview
********

XGBoost is designed to be an extensible library.  One way to extend it is by providing our
own objective function for training and corresponding metric for performance monitoring.
This document introduces implementing a customized elementwise evaluation metric and
objective for XGBoost.  Although the introduction uses Python for demonstration, the
concepts should be readily applicable to other language bindings.

.. note::

   * The ranking task does not support customized functions.
   * Breaking change was made in XGBoost 1.6.

In the following two sections, we will provide a step by step walk through of implementing
the ``Squared Log Error (SLE)`` objective function:

.. math::
   \frac{1}{2}[log(pred + 1) - log(label + 1)]^2

and its default metric ``Root Mean Squared Log Error(RMSLE)``:

.. math::
   \sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}

Although XGBoost has native support for said functions, using it for demonstration
provides us the opportunity of comparing the result from our own implementation and the
one from XGBoost internal for learning purposes.  After finishing this tutorial, we should
be able to provide our own functions for rapid experiments.  And at the end, we will
provide some notes on non-identy link function along with examples of using custom metric
and objective with `scikit-learn` interface.
with scikit-learn interface.

*****************************
Customized Objective Function
*****************************

During model training, the objective function plays an important role: provide gradient
information, both first and second order gradient, based on model predictions and observed
data labels (or targets).  Therefore, a valid objective function should accept two inputs,
namely prediction and labels.  For implementing ``SLE``, we define:

.. code-block:: python

    import numpy as np
    import xgboost as xgb
    from typing import Tuple

    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        y = dtrain.get_label()
        return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''
        y = dtrain.get_label()
        return ((-np.log1p(predt) + np.log1p(y) + 1) /
                np.power(predt + 1, 2))

    def squared_log(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.
        '''
        predt[predt < -1] = -1 + 1e-6
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess


In the above code snippet, ``squared_log`` is the objective function we want.  It accepts a
numpy array ``predt`` as model prediction, and the training DMatrix for obtaining required
information, including labels and weights (not used here).  This objective is then used as
a callback function for XGBoost during training by passing it as an argument to
``xgb.train``:

.. code-block:: python

   xgb.train({'tree_method': 'hist', 'seed': 1994},  # any other tree method is fine.
              dtrain=dtrain,
              num_boost_round=10,
              obj=squared_log)

Notice that in our definition of the objective, whether we subtract the labels from the
prediction or the other way around is important.  If you find the training error goes up
instead of down, this might be the reason.


**************************
Customized Metric Function
**************************

So after having a customized objective, we might also need a corresponding metric to
monitor our model's performance.  As mentioned above, the default metric for ``SLE`` is
``RMSLE``.  Similarly we define another callback like function as the new metric:

.. code-block:: python

    def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        ''' Root mean squared log error metric.'''
        y = dtrain.get_label()
        predt[predt < -1] = -1 + 1e-6
        elements = np.power(np.log1p(y) - np.log1p(predt), 2)
        return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))

Since we are demonstrating in Python, the metric or objective need not be a function,
any callable object should suffice.  Similar to the objective function, our metric also
accepts ``predt`` and ``dtrain`` as inputs, but returns the name of the metric itself and a
floating point value as the result.  After passing it into XGBoost as argument of ``feval``
parameter:

.. code-block:: python

    xgb.train({'tree_method': 'hist', 'seed': 1994,
               'disable_default_eval_metric': 1},
              dtrain=dtrain,
              num_boost_round=10,
              obj=squared_log,
              feval=rmsle,
              evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
              evals_result=results)

We will be able to see XGBoost printing something like:

.. code-block:: none

    [0] dtrain-PyRMSLE:1.37153  dtest-PyRMSLE:1.31487
    [1] dtrain-PyRMSLE:1.26619  dtest-PyRMSLE:1.20899
    [2] dtrain-PyRMSLE:1.17508  dtest-PyRMSLE:1.11629
    [3] dtrain-PyRMSLE:1.09836  dtest-PyRMSLE:1.03871
    [4] dtrain-PyRMSLE:1.03557  dtest-PyRMSLE:0.977186
    [5] dtrain-PyRMSLE:0.985783 dtest-PyRMSLE:0.93057
    ...

Notice that the parameter ``disable_default_eval_metric`` is used to suppress the default metric
in XGBoost.

For fully reproducible source code and comparison plots, see
:ref:`sphx_glr_python_examples_custom_rmsle.py`.

*********************
Reverse Link Function
*********************

When using builtin objective, the raw prediction is transformed according to the objective
function.  When a custom objective is provided XGBoost doesn't know its link function so the
user is responsible for making the transformation for both objective and custom evaluation
metric.  For objective with identiy link like ``squared error`` this is trivial, but for
other link functions like log link or inverse link the difference is significant.

For the Python package, the behaviour of prediction can be controlled by the
``output_margin`` parameter in ``predict`` function.  When using the ``custom_metric``
parameter without a custom objective, the metric function will receive transformed
prediction since the objective is defined by XGBoost. However, when the custom objective is
also provided along with that metric, then both the objective and custom metric will
recieve raw prediction.  The following example provides a comparison between two different
behavior with a multi-class classification model. Firstly we define 2 different Python
metric functions implementing the same underlying metric for comparison,
`merror_with_transform` is used when custom objective is also used, otherwise the simpler
`merror` is preferred since XGBoost can perform the transformation itself.

.. code-block:: python

    import xgboost as xgb
    import numpy as np

    def merror_with_transform(predt: np.ndarray, dtrain: xgb.DMatrix):
        """Used when custom objective is supplied."""
        y = dtrain.get_label()
        n_classes = predt.size // y.shape[0]
        # Like custom objective, the predt is untransformed leaf weight when custom objective
        # is provided.

        # With the use of `custom_metric` parameter in train function, custom metric receives
        # raw input only when custom objective is also being used.  Otherwise custom metric
        # will receive transformed prediction.
        assert predt.shape == (d_train.num_row(), n_classes)
        out = np.zeros(dtrain.num_row())
        for r in range(predt.shape[0]):
            i = np.argmax(predt[r])
            out[r] = i

        assert y.shape == out.shape

        errors = np.zeros(dtrain.num_row())
        errors[y != out] = 1.0
        return 'PyMError', np.sum(errors) / dtrain.num_row()

The above function is only needed when we want to use custom objective and XGBoost doesn't
know how to transform the prediction.  The normal implementation for multi-class error
function is:

.. code-block:: python

    def merror(predt: np.ndarray, dtrain: xgb.DMatrix):
        """Used when there's no custom objective."""
        # No need to do transform, XGBoost handles it internally.
        errors = np.zeros(dtrain.num_row())
        errors[y != out] = 1.0
        return 'PyMError', np.sum(errors) / dtrain.num_row()


Next we need the custom softprob objective:

.. code-block:: python

    def softprob_obj(predt: np.ndarray, data: xgb.DMatrix):
        """Loss function.  Computing the gradient and approximated hessian (diagonal).
        Reimplements the `multi:softprob` inside XGBoost.
        """

        # Full implementation is available in the Python demo script linked below
        ...

        return grad, hess

Lastly we can train the model using ``obj`` and ``custom_metric`` parameters:

.. code-block:: python

    Xy = xgb.DMatrix(X, y)
    booster = xgb.train(
        {"num_class": kClasses, "disable_default_eval_metric": True},
        m,
        num_boost_round=kRounds,
        obj=softprob_obj,
        custom_metric=merror_with_transform,
        evals_result=custom_results,
        evals=[(m, "train")],
    )

Or if you don't need the custom objective and just want to supply a metric that's not
available in XGBoost:

.. code-block:: python

    booster = xgb.train(
        {
            "num_class": kClasses,
            "disable_default_eval_metric": True,
            "objective": "multi:softmax",
        },
        m,
        num_boost_round=kRounds,
        # Use a simpler metric implementation.
        custom_metric=merror,
        evals_result=custom_results,
        evals=[(m, "train")],
    )

We use ``multi:softmax`` to illustrate the differences of transformed prediction.  With
``softprob`` the output prediction array has shape ``(n_samples, n_classes)`` while for
``softmax`` it's ``(n_samples, )``. A demo for multi-class objective function is also
available at :ref:`sphx_glr_python_examples_custom_softmax.py`.


**********************
Scikit-Learn Interface
**********************


The scikit-learn interface of XGBoost has some utilities to improve the integration with
standard scikit-learn functions.  For instance, after XGBoost 1.6.0 users can use the cost
function (not scoring functions) from scikit-learn out of the box:

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_absolute_error
    X, y = load_diabetes(return_X_y=True)
    reg = xgb.XGBRegressor(
        tree_method="hist",
        eval_metric=mean_absolute_error,
    )
    reg.fit(X, y, eval_set=[(X, y)])

Also, for custom objective function, users can define the objective without having to
access ``DMatrix``:

.. code-block:: python

    def softprob_obj(labels: np.ndarray, predt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = labels.shape[0]
        classes = predt.shape[1]
        grad = np.zeros((rows, classes), dtype=float)
        hess = np.zeros((rows, classes), dtype=float)
        eps = 1e-6
        for r in range(predt.shape[0]):
            target = labels[r]
            p = softmax(predt[r, :])
            for c in range(predt.shape[1]):
                g = p[c] - 1.0 if c == target else p[c]
                h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
                grad[r, c] = g
                hess[r, c] = h

        grad = grad.reshape((rows * classes, 1))
        hess = hess.reshape((rows * classes, 1))
        return grad, hess

    clf = xgb.XGBClassifier(tree_method="hist", objective=softprob_obj)
