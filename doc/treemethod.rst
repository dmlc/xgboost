####################
XGBoost Tree Methods
####################

For training boosted tree models, there are 2 parameters used for choosing algorithms,
namely ``updater`` and ``tree_method``.  XGBoost has 4 builtin tree methods, namely
``exact``, ``approx``, ``hist`` and ``gpu_hist``.  Along with these tree methods, there
are also some free standing updaters including ``grow_local_histmaker``, ``refresh``,
``prune`` and ``sync``.  The parameter ``updater`` is more primitive than ``tree_method``
as the latter is just a pre-configuration of the former.  The difference is mostly due to
historical reasons that each updater requires some specific configurations and might has
missing features.  As we are moving forward, the gap between them is becoming more and
more irrevelant.  We will collectively document them under tree methods.

**************
Exact Solution
**************

Exact means XGBoost considers all candidates from data for tree splitting, but underlying
the objective is still interpreted as a Taylor expansion.

1. ``exact``: Vanilla tree boosting tree algorithm described in `reference paper
   <http://arxiv.org/abs/1603.02754>`_.  During each split finding procedure, it iterates
   over every entry of input data.  It's more accurate (among other greedy methods) but
   slow in computation performance.  Also it doesn't support distributed training as
   XGBoost employs row spliting data distribution while ``exact`` tree method works on a
   sorted column format.  This tree method can be used with parameter ``tree_method`` set
   to ``exact``.


**********************
Approximated Solutions
**********************

As ``exact`` tree method is slow in performance and not scalable, we often employ
approximated training algorithms.  These algorithms build a gradient histogram for each
node and iterate through the histogram instead of real dataset.  Here we introduce the
implementations in XGBoost below.

1. ``grow_local_histmaker`` updater: An approximation tree method described in `reference
   paper <http://arxiv.org/abs/1603.02754>`_.  This updater is rarely used in practice so
   it's still an updater rather than tree method.  During split finding, it first runs a
   weighted GK sketching for data points belong to current node to find split candidates,
   using hessian as weights.  The histogram is built upon this per-node sketch.  It's
   faster than ``exact`` in some applications, but still slow in computation.

2. ``approx`` tree method: An approximation tree method described in `reference paper
   <http://arxiv.org/abs/1603.02754>`_.  Different from ``grow_local_histmaker``, it runs
   sketching before building each tree using all the rows (rows belonging to the root)
   instead of per-node dataset.  Similar to ``grow_local_histmaker`` updater, hessian is
   used as weights during sketch.  The algorithm can be accessed by setting
   ``tree_method`` to ``approx``.

3. ``hist`` tree method: An approximation tree method used in LightGBM with slight
   differences in implementation.  It runs sketching before training using only user
   provided weights instead of hessian.  The subsequent per-node histogram is built upon
   this global sketch.  This is the fastest algorithm as it runs sketching only once.  The
   algorithm can be accessed by setting ``tree_method`` to ``hist``.

4. ``gpu_hist`` tree method: The ``gpu_hist`` tree method is a GPU implementation of
   ``hist``, with additional support for gradient based sampling.  The algorithm can be
   accessed by setting ``tree_method`` to ``gpu_hist``.

************
Implications
************

Some objectives like ``reg:squarederror`` have constant hessian.  In this case, ``hist``
or ``gpu_hist`` should be preferred as weighted sketching doesn't make sense with constant
weights.  When using non-constant hessian objectives, sometimes ``approx`` yields better
accuracy, but with slower computation performance.  Most of the time using ``(gpu)_hist``
with higher ``max_bin`` can achieve similar or even superior accuracy while maintaining
good performance.  However, as xgboost is largely driven by community effort, the actual
implementations have some differences than pure math description.  Result might have
slight differences than expectation, which we are currently trying to overcome.

**************
Other Updaters
**************

1. ``Pruner``: It prunes the built tree by ``gamma`` parameter.  ``pruner`` is usually
   used as part of other tree methods.
2. ``Refresh``: Refresh the statistic of bulilt trees on a new training dataset.
3. ``Sync``: Synchronize the tree among workers when running distributed training.

****************
Removed Updaters
****************

2 Updaters were removed during development due to maintainability.  We describe them here
solely for the interest of documentation.  First one is distributed colmaker, which was a
distributed version of exact tree method.  It required specialization for column based
spliting strategy and a different prediction procedure.  As the exact tree method is slow
by itself and scaling is even less efficient, we removed it entirely.  Second one is
``skmaker``.  Per-node weighted sketching employed by ``grow_local_histmaker`` is slow,
the ``skmaker`` was unmaintained and seems to be a workaround trying to eliminate the
histogram creation step and uses sketching values directly during split evaluation.  It
was never tested and contained some unknown bugs, we decided to remove it and focus our
resources on more promising algorithms instead.  For accuracy, most of the time
``approx``, ``hist`` and ``gpu_hist`` are enough with some parameters tunning, so removing
them don't have any real practical impact.
