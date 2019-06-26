######################################
Custom Objective and Evaluation Metric
######################################

XGBoost is designed to be an extensible library.  One way to extend it is by providing our
own objective function for training and corresponding metric for performance monitoring.
This document introduces implementing a customized elementwise evaluation metric and
objective for XGBoost.  Although the introduction uses Python for demonstration, the
concepts should be readily applicable to other language bindings.

.. note::

   * The ranking task does not support customized functions.
   * The customized functions defined here are only applicable to single node training.
     Distributed environment requires syncing with ``xgboost.rabit``, the interface is
     subject to change hence beyond the scope of this tutorial.
   * We also plan to re-design the interface for multi-classes objective in the future.

In the following sections, we will provide a step by step walk through of implementing
``Squared Log Error(SLE)`` objective function:

.. math::
   \frac{1}{2}[log(pred + 1) - log(label + 1)]^2

and its default metric ``Root Mean Squared Log Error(RMSLE)``:

.. math::
   \sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}

Although XGBoost has native support for said functions, using it for demonstration
provides us the opportunity of comparing the result from our own implementation and the
one from XGBoost internal for learning purposes.  After finishing this tutorial, we should
be able to provide our own functions for rapid experiments.

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

Since we are demonstrating in Python, the metric or objective needs not be a function,
any callable object should suffice.  Similarly to the objective function, our metric also
accepts ``predt`` and ``dtrain`` as inputs, but returns the name of metric itself and a
floating point value as result.  After passing it into XGBoost as argument of ``feval``
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

    [0]	dtrain-PyRMSLE:1.37153	dtest-PyRMSLE:1.31487
    [1]	dtrain-PyRMSLE:1.26619	dtest-PyRMSLE:1.20899
    [2]	dtrain-PyRMSLE:1.17508	dtest-PyRMSLE:1.11629
    [3]	dtrain-PyRMSLE:1.09836	dtest-PyRMSLE:1.03871
    [4]	dtrain-PyRMSLE:1.03557	dtest-PyRMSLE:0.977186
    [5]	dtrain-PyRMSLE:0.985783	dtest-PyRMSLE:0.93057
    ...

Notice that the parameter ``disable_default_eval_metric`` is used to suppress the default metric
in XGBoost.

For fully reproducible source code and comparison plots, see `custom_rmsle.py <https://github.com/dmlc/xgboost/tree/master/demo/guide-python/custom_rmsle.py>`_.
