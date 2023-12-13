#########
Intercept
#########

.. versionadded:: 2.0.0

Since 2.0.0, XGBoost supports estimating the model intercept (named ``base_score``)
automatically based on targets upon training. The behavior can be controlled by setting
``base_score`` to a constant value. The following snippet disables the automatic
estimation:

.. code-block:: python

    import xgboost as xgb

    reg = xgb.XGBRegressor()
    reg.set_params(base_score=0.5)

In addition, here 0.5 represents the value after applying the inverse link function. See
the end of the document for a description.

Other than the ``base_score``, users can also provide global bias via the data field
``base_margin``, which is a vector or a matrix depending on the task.

.. code-block:: python

    import xgboost as xgb
    from sklearn.datasets import make_regression

    X, y = make_regression()

    reg = xgb.XGBRegressor()
    reg.fit(X, y)
    # Request for raw prediction
    m = reg.predict(X, output_margin=True)

    reg_1 = xgb.XGBRegressor()
    # Feed the prediction into the next model
    reg.fit(X, y, base_margin=m)
    reg.predict(X, base_margin=m)


It specifies the bias for each sample and can be used for stacking an XGBoost model on top
of other models, see :ref:`sphx_glr_python_examples_boost_from_prediction.py` for a worked
example. When ``base_margin`` is specified, it automatically overrides the ``base_score``
parameter. If you are stacking XGBoost models, then the usage should be relatively
straightforward, with the previous model providing raw prediction and a new model using
the prediction as bias. For more customized inputs, users need to take extra care of the
link function. Let :math:`F` be the model and :math:`g` be the link function, since
``base_score`` is overridden when sample-specific ``base_margin`` is available, we will
omit it here:

.. math::

   g(E[y_i]) = F(x_i)


When base margin :math:`b` is provided, it's added to the raw model output :math:`F`:

.. math::

   g(E[y_i]) = F(x_i) + b_i

and the output of the final model is:


.. math::

   g^{-1}(F(x_i) + b_i)

Using the gamma deviance objective ``reg:gamma`` as an example, which has a log link
function, hence:

.. math::

   \ln{(E[y_i])} = F(x_i) + b_i \\
   E[y_i] = \exp{(F(x_i) + b_i)}

As a result, if you are feeding outputs from models like GLM with a corresponding
objective function, make sure the outputs are not yet transformed by the inverse link.

In the case of ``base_score`` (intercept), if you access the estimation through
:py:meth:`~xgboost.Booster.save_config`, XGBoost returns the value
:math:`g^{-1}(base_score)`. With logistic regression and the logit link function, given
the ``base_score`` as 0.5, :math:`logit(0.5) = 0.0` is added to the raw model output:

.. math::

   E[y_i] = g^{-1}{(F(x_i) + g(intercept))}

This is more intuitive if you remove the model and consider only the intercept, which is
estimated before the model is fitted:

.. math::

   E[y_i] = g^{-1}{g(intercept))} \\
   E[y_i] = intercept