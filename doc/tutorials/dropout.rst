#######
Dropout
#######

XGBoost supports expectation-preserving dropout for tree boosters through
``dropout_rate``. It is configured directly on the default ``gbtree`` booster.
The legacy ``booster=dart`` value remains available as a deprecated alias.

************
How it works
************

Let the prediction before round :math:`m` be
:math:`F(x) = F_0(x) + \sum_{i=1}^{m-1} F_i(x)`, where :math:`F_0` is the
base score or base margin. For dropout probability :math:`p`, independently sample
:math:`I_i \sim \operatorname{Bernoulli}(1-p)` and compute gradients from

.. math::

  \widetilde{F}(x) = F_0(x) + \sum_{i=1}^{m-1} \frac{I_i}{1-p} F_i(x).

Since :math:`\mathbb{E}[I_i/(1-p)] = 1`, the temporary prediction does not
over- or undershoot the full ensemble in expectation:

.. math::

  \mathbb{E}[\widetilde{F}(x)] = F(x).

The new tree is fitted to gradients computed from :math:`\widetilde{F}` and then
committed normally with its learning-rate-scaled leaf values. Existing and new trees
are never reweighted. Consequently, saved models and inference use the ordinary
additive-tree path and require no dropout-specific work.

For squared error, the gradient is affine in the margin, so its expectation is exactly
the ordinary boosting gradient. For nonlinear objectives, the margin remains unbiased
but the expected gradient can differ.

Because a fresh random mask is used for every training prediction, training can be
slower than ordinary boosting and early stopping can be noisier.

************************
Relation to row sampling
************************

``dropout_rate`` is not directly comparable to ``1 - subsample``. Row sampling changes
only the tree fitted in the current round, whereas prediction dropout perturbs the
accumulated ensemble and can have a stronger effect as the ensemble grows. When
comparing the two forms of regularization, start with a ``dropout_rate`` lower than the
omitted-row fraction and tune it independently together with the learning rate.

**********
Parameters
**********

* ``dropout_rate``: probability of independently dropping each existing tree before
  gradient computation. The valid range is ``[0.0, 1.0)`` and the default is ``0.0``.

``skip_drop`` is accepted as a deprecated alias for ``dropout_rate``. When both are
specified, ``dropout_rate`` takes precedence. ``sample_type``, ``normalize_type``,
``rate_drop``, and ``one_drop`` are accepted temporarily but ignored, with warnings.

*************
Sample Script
*************

.. code-block:: python

  import xgboost as xgb

  dtrain = xgb.DMatrix("demo/data/agaricus.txt.train?format=libsvm")
  dtest = xgb.DMatrix("demo/data/agaricus.txt.test?format=libsvm")
  params = {
      "max_depth": 5,
      "learning_rate": 0.1,
      "objective": "binary:logistic",
      "dropout_rate": 0.05,
  }
  bst = xgb.train(params, dtrain, num_boost_round=50)
  preds = bst.predict(dtest)
