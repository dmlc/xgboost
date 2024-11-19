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
``base_margin``, which is a vector or a matrix depending on the task. With multi-output
and multi-class, the ``base_margin`` is a matrix with size ``(n_samples, n_targets)`` or
``(n_samples, n_classes)``.

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
    reg_1.fit(X, y, base_margin=m)
    reg_1.predict(X, base_margin=m)


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
objective function, make sure the outputs are not yet transformed by the inverse link
(activation).

In the case of ``base_score`` (intercept), it can be accessed through
:py:meth:`~xgboost.Booster.save_config` after estimation. Unlike the ``base_margin``, the
returned value represents a value after applying inverse link.  With logistic regression
and the logit link function as an example, given the ``base_score`` as 0.5,
:math:`g(intercept) = logit(0.5) = 0` is added to the raw model output:

.. math::

   E[y_i] = g^{-1}{(F(x_i) + g(intercept))}

and 0.5 is the same as :math:`base\_score = g^{-1}(0) = 0.5`. This is more intuitive if
you remove the model and consider only the intercept, which is estimated before the model
is fitted:

.. math::

   E[y] = g^{-1}{(g(intercept))} \\
   E[y] = intercept

For some objectives like MAE, there are close solutions, while for others it's estimated
with one step Newton method.

******
Offset
******

The ``base_margin`` is a form of ``offset`` in GLM. Using the Poisson objective as an
example, we might want to model the rate instead of the count:

.. math::

   rate = \frac{count}{exposure}

And the offset is defined as log link applied to the exposure variable:
:math:`\ln{exposure}`. Let :math:`c` be the count and :math:`\gamma` be the exposure,
substituting the response :math:`y` in our previous formulation of base margin:

.. math::

   g(\frac{E[c_i]}{\gamma_i}) = F(x_i)

Substitute :math:`g` with :math:`\ln` for Poisson regression:

.. math::

   \ln{\frac{E[c_i]}{\gamma_i}} = F(x_i)

We have:

.. math::

   E[c_i] &= \exp{(F(x_i) + \ln{\gamma_i})} \\
   E[c_i] &= g^{-1}(F(x_i) + g(\gamma_i))

As you can see, we can use the ``base_margin`` for modeling with offset similar to GLMs