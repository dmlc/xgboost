#########
Intercept
#########

.. versionadded:: 2.0.0

Since 2.0.0, XGBoost supports estimating the model intercept (named ``base_score``)
automatically based on targets upon training. The behavior can be controlled by setting
``base_score`` to a constant value. The following snippet disables the automatic
estimation:

.. tabs::
    .. code-tab:: py

        import xgboost as xgb

        clf = xgb.XGBClassifier(n_estimators=10)
        clf.set_params(base_score=0.5)

    .. code-tab:: r R

        library(xgboost)

        # Load built-in dataset
        data(agaricus.train, package = "xgboost")

        # Set base_score parameter directly
        model <- xgboost(
          x = agaricus.train$data,
          y = factor(agaricus.train$label),
          base_score = 0.5,
          nrounds = 10
        )

In addition, here 0.5 represents the value after applying the inverse link function. See
the end of the document for a description.

Other than the ``base_score``, users can also provide global bias via the data field
``base_margin``, which is a vector or a matrix depending on the task. With multi-output
and multi-class, the ``base_margin`` is a matrix with size ``(n_samples, n_targets)`` or
``(n_samples, n_classes)``.

.. tabs::
    .. code-tab:: py

        import xgboost as xgb
        from sklearn.datasets import make_classification

        X, y = make_classification()

        clf = xgb.XGBClassifier()
        clf.fit(X, y)
        # Request for raw prediction
        m = clf.predict(X, output_margin=True)

        clf_1 = xgb.XGBClassifier()
        # Feed the prediction into the next model
        # Using base margin overrides the base score, see below sections.
        clf_1.fit(X, y, base_margin=m)
        clf_1.predict(X, base_margin=m)

    .. code-tab:: r R

        library(xgboost)

        # Load built-in dataset
        data(agaricus.train, package = "xgboost")

        # Train first model
        model_1 <- xgboost(
          x = agaricus.train$data,
          y = factor(agaricus.train$label),
          nrounds = 10
        )

        # Request for raw prediction
        m <- predict(model_1, agaricus.train$data, type = "raw")

        # Feed the prediction into the next model using base_margin
        # Using base margin overrides the base score, see below sections.
        model_2 <- xgboost(
          x = agaricus.train$data,
          y = factor(agaricus.train$label),
          base_margin = m,
          nrounds = 10
        )

        # Make predictions with base_margin
        pred <- predict(model_2, agaricus.train$data, base_margin = m)


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

*******
Example
*******

The following example shows the relationship between ``base_score`` and ``base_margin``
using binary logistic with a `logit` link function:

.. tabs::
    .. code-tab:: py

        import numpy as np
        from scipy.special import logit
        from sklearn.datasets import make_classification

        import xgboost as xgb

        X, y = make_classification(random_state=2025)

    .. code-tab:: r R

        library(xgboost)

        # Load built-in dataset
        data(agaricus.train, package = "xgboost")
        X <- agaricus.train$data
        y <- agaricus.train$label

The intercept is a valid probability (0.5). It's used as the initial estimation of the
probability of obtaining a positive sample.

.. tabs::
    .. code-tab:: py

        intercept = 0.5

    .. code-tab:: r R

        intercept <- 0.5

First we use the intercept to train a model:

.. tabs::
    .. code-tab:: py

        booster = xgb.train(
            {"base_score": intercept, "objective": "binary:logistic"},
            dtrain=xgb.DMatrix(X, y),
            num_boost_round=1,
        )
        predt_0 = booster.predict(xgb.DMatrix(X, y))

    .. code-tab:: r R

        # First model with base_score
        model_0 <- xgboost(
          x = X, y = factor(y),
          base_score = intercept,
          objective = "binary:logistic",
          nrounds = 1
        )
        predt_0 <- predict(model_0, X)

Apply :py:func:`~scipy.special.logit` to obtain the "margin":

.. tabs::
    .. code-tab:: py

        # Apply logit function to obtain the "margin"
        margin = np.full(y.shape, fill_value=logit(intercept), dtype=np.float32)
        Xy = xgb.DMatrix(X, y, base_margin=margin)
        # Second model with base_margin
        # 0.2 is a dummy value to show that `base_margin` overrides `base_score`.
        booster = xgb.train(
            {"base_score": 0.2, "objective": "binary:logistic"},
            dtrain=Xy,
            num_boost_round=1,
        )
        predt_1 = booster.predict(Xy)

    .. code-tab:: r R

        # Apply logit function to obtain the "margin"
        logit_intercept <- log(intercept / (1 - intercept))
        margin <- rep(logit_intercept, length(y))
        # Second model with base_margin
        # 0.2 is a dummy value to show that `base_margin` overrides `base_score`
        model_1 <- xgboost(
          x = X, y = factor(y),
          base_margin = margin,
          base_score = 0.2,
          objective = "binary:logistic",
          nrounds = 1
        )
        predt_1 <- predict(model_1, X, base_margin = margin)

Compare the results:

.. tabs::
    .. code-tab:: py

        np.testing.assert_allclose(predt_0, predt_1)

    .. code-tab:: r R

        all.equal(predt_0, predt_1, tolerance = 1e-6)
