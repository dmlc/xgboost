################
Tree subsampling
################

XGBoost supports expectation-preserving subsampling of existing trees through
``tree_subsample``. It is configured directly on the default ``gbtree`` booster.
The legacy ``booster=dart`` value remains available as a deprecated alias.

************
How it works
************

Let the prediction before round :math:`m` be
:math:`F(x) = F_0(x) + \sum_{i=1}^{m-1} F_i(x)`, where :math:`F_0` is the
base score or base margin. For tree retention probability :math:`q`, independently
sample :math:`I_i \sim \operatorname{Bernoulli}(q)` and compute gradients from

.. math::

  \widetilde{F}(x) = F_0(x) + \sum_{i=1}^{m-1} \frac{I_i}{q} F_i(x).

Since :math:`\mathbb{E}[I_i/q] = 1`, the temporary prediction does not over- or
undershoot the full ensemble in expectation:

.. math::

  \mathbb{E}[\widetilde{F}(x)] = F(x).

The new tree is fitted to gradients computed from :math:`\widetilde{F}` and then
committed normally with its learning-rate-scaled leaf values. Existing and new trees
are never reweighted. Consequently, saved models and inference use the ordinary
additive-tree path and require no tree-subsampling-specific work.

************************
Relation to row sampling
************************

Tree subsampling introduces variance into training. Like row subsampling, this can
help reduce overfitting. However, row subsampling perturbs only the tree fitted in the
current round, so its relative influence on the total ensemble prediction diminishes as
the ensemble grows. Tree subsampling instead perturbs every accumulated tree at the
same per-tree probability, so its effect does not vanish merely because the ensemble
contains more trees. Consequently, the omitted-tree fraction ``1 - tree_subsample`` will
often need to be smaller than the omitted-row fraction ``1 - subsample``; equivalently,
``tree_subsample`` will often be closer to ``1`` than ``subsample``.

**********
Parameters
**********

* ``tree_subsample``: probability of independently retaining each existing tree before
  gradient computation. The valid range is ``[0.000001, 1.0]`` and the default is ``1.0``.

The legacy ``rate_drop`` parameter is accepted temporarily. When ``tree_subsample`` is
not supplied, ``rate_drop=r`` is converted to ``tree_subsample=1-r`` with a warning.
This preserves the uniform tree-retention probability, but it does not preserve the
legacy DART normalization algorithm, so training behavior will differ. If both
parameters are supplied, ``tree_subsample`` takes precedence.

``sample_type``, ``normalize_type``, ``one_drop``, and ``skip_drop`` have no exact
conversion and are ignored with removal warnings. See the
`DART removal discussion <https://github.com/dmlc/xgboost/issues/12339>`_ for details.

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
      "tree_subsample": 0.95,
  }
  bst = xgb.train(params, dtrain, num_boost_round=50)
  preds = bst.predict(dtest)
