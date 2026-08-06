#######
Dropout
#######

XGBoost supports dropout for tree boosters through ``dropout_rate``. Dropout is
configured directly on the default tree booster.

Tree dropout is inspired by the original paper by Rashmi Korlakai Vinayak and
Ran Gilad-Bachrach (`PMLR <http://proceedings.mlr.press/v38/korlakaivinayak15.pdf>`_,
`arXiv <https://arxiv.org/abs/1505.01866>`_). XGBoost uses a simplified normalization
that acts on the temporary training prediction instead of changing committed tree weights.

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
overshoot or undershoot the full ensemble in expectation:

.. math::

  \mathbb{E}[\widetilde{F}(x)] = F(x).

The new tree is fitted to gradients computed from :math:`\widetilde{F}` and then
committed with its ordinary learning-rate-scaled leaf values. No existing or new tree
is reweighted. Consequently, saved models and inference use the ordinary additive-tree
path and require no dropout-specific work.

For squared error, the gradient is affine in the margin, so its expectation is exactly
the ordinary boosting gradient. For nonlinear objectives, the margin remains unbiased
but the expected gradient can differ.

Because a fresh random mask is used for each training prediction, training metrics and
early stopping can be noisier than with ordinary boosting.

**********
Parameters
**********

* ``dropout_rate``: probability of independently dropping each existing tree before
  gradient computation. The valid range is ``[0.0, 1.0)`` and the default is ``0.0``.

``skip_drop`` is accepted as an alias for ``dropout_rate`` with a removal warning.
``sample_type``, ``normalize_type``, ``rate_drop``, and ``one_drop`` have been removed;
they are ignored and emit removal warnings.

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
      "dropout_rate": 0.1,
  }
  bst = xgb.train(params, dtrain, num_boost_round=50)
  preds = bst.predict(dtest)
