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

************************
Relation to row sampling
************************

Dropout introduces variance into training. Like row subsampling, this can help reduce
overfitting. However, subsampling perturbs only the tree fitted in the current round, so
its relative influence on the total ensemble prediction diminishes as the ensemble
grows. Dropout instead perturbs every accumulated tree at the same per-tree probability,
so its effect does not vanish merely because the ensemble contains more trees.
Consequently, ``dropout_rate`` will often need to be smaller than the omitted-row
fraction ``1 - subsample`` and should be tuned independently.

**********
Parameters
**********

* ``dropout_rate``: probability of independently dropping each existing tree before
  gradient computation. The valid range is ``[0.0, 0.999999]`` and the default is ``0.0``.

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
