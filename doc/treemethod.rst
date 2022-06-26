############
Tree Methods
############

For training boosted tree models, there are 2 parameters used for choosing algorithms,
namely ``updater`` and ``tree_method``.  XGBoost has 4 builtin tree methods, namely
``exact``, ``approx``, ``hist`` and ``gpu_hist``.  Along with these tree methods, there
are also some free standing updaters including ``refresh``,
``prune`` and ``sync``.  The parameter ``updater`` is more primitive than ``tree_method``
as the latter is just a pre-configuration of the former.  The difference is mostly due to
historical reasons that each updater requires some specific configurations and might has
missing features.  As we are moving forward, the gap between them is becoming more and
more irrelevant.  We will collectively document them under tree methods.

**************
Exact Solution
**************

Exact means XGBoost considers all candidates from data for tree splitting, but underlying
the objective is still interpreted as a Taylor expansion.

1. ``exact``: Vanilla gradient boosting tree algorithm described in `reference paper
   <http://arxiv.org/abs/1603.02754>`_.  During each split finding procedure, it iterates
   over all entries of input data.  It's more accurate (among other greedy methods) but
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

1. ``approx`` tree method: An approximation tree method described in `reference paper
   <http://arxiv.org/abs/1603.02754>`_.  It runs sketching before building each tree
   using all the rows (rows belonging to the root). Hessian is used as weights during
   sketch.  The algorithm can be accessed by setting ``tree_method`` to ``approx``.

2. ``hist`` tree method: An approximation tree method used in LightGBM with slight
   differences in implementation.  It runs sketching before training using only user
   provided weights instead of hessian.  The subsequent per-node histogram is built upon
   this global sketch.  This is the fastest algorithm as it runs sketching only once.  The
   algorithm can be accessed by setting ``tree_method`` to ``hist``.

3. ``gpu_hist`` tree method: The ``gpu_hist`` tree method is a GPU implementation of
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

1. ``Prune``: It prunes the existing trees.  ``prune`` is usually used as part of other
   tree methods.  To use pruner independently, one needs to set the process type to update
   by: ``{"process_type": "update", "updater": "prune"}``.  With this set of parameters,
   during trianing, XGBOost will prune the existing trees according to 2 parameters
   ``min_split_loss (gamma)`` and ``max_depth``.

2. ``Refresh``: Refresh the statistic of built trees on a new training dataset.  Like the
   pruner, To use refresh independently, one needs to set the process type to update:
   ``{"process_type": "update", "updater": "refresh"}``.  During training, the updater
   will change statistics like ``cover`` and ``weight`` according to the new training
   dataset.  When ``refresh_leaf`` is also set to true (default), XGBoost will update the
   leaf value according to the new leaf weight, but the tree structure (split condition)
   itself doesn't change.

   There are examples on both training continuation (adding new trees) and using update
   process on ``demo/guide-python``.  Also checkout the ``process_type`` parameter in
   :doc:`parameter`.

3. ``Sync``: Synchronize the tree among workers when running distributed training.

****************
Removed Updaters
****************

3 Updaters were removed during development due to maintainability.  We describe them here
solely for the interest of documentation.

1. Distributed colmaker, which was a distributed version of exact tree method.  It
   required specialization for column based splitting strategy and a different prediction
   procedure.  As the exact tree method is slow by itself and scaling is even less
   efficient, we removed it entirely.

2. ``skmaker``.  Per-node weighted sketching employed by ``grow_local_histmaker`` is slow,
   the ``skmaker`` was unmaintained and seems to be a workaround trying to eliminate the
   histogram creation step and uses sketching values directly during split evaluation.  It
   was never tested and contained some unknown bugs, we decided to remove it and focus our
   resources on more promising algorithms instead.  For accuracy, most of the time
   ``approx``, ``hist`` and ``gpu_hist`` are enough with some parameters tuning, so
   removing them don't have any real practical impact.

3. ``grow_local_histmaker`` updater: An approximation tree method described in `reference
   paper <http://arxiv.org/abs/1603.02754>`_.  This updater was rarely used in practice so
   it was still an updater rather than tree method.  During split finding, it first runs a
   weighted GK sketching for data points belong to current node to find split candidates,
   using hessian as weights.  The histogram is built upon this per-node sketch.  It was
   faster than ``exact`` in some applications, but still slow in computation.  It was
   removed because it depended on Rabit's customized reduction function that handles all
   the data structure that can be serialized/deserialized into fixed size buffer, which is
   not directly supported by NCCL or federated learning gRPC, making it hard to refactor
   into a common allreducer interface.

**************
Feature Matrix
**************

Following table summarizes some differences in supported features between 4 tree methods,
`T` means supported while `F` means unsupported.

+------------------+-----------+---------------------+---------------------+------------------------+
|                  | Exact     | Approx              | Hist                | GPU Hist               |
+==================+===========+=====================+=====================+========================+
| grow_policy      | Depthwise | depthwise/lossguide | depthwise/lossguide | depthwise/lossguide    |
+------------------+-----------+---------------------+---------------------+------------------------+
| max_leaves       | F         | T                   | T                   | T                      |
+------------------+-----------+---------------------+---------------------+------------------------+
| sampling method  | uniform   | uniform             | uniform             | gradient_based/uniform |
+------------------+-----------+---------------------+---------------------+------------------------+
| categorical data | F         | T                   | T                   | T                      |
+------------------+-----------+---------------------+---------------------+------------------------+
| External memory  | F         | T                   | T                   | P                      |
+------------------+-----------+---------------------+---------------------+------------------------+
| Distributed      | F         | T                   | T                   | T                      |
+------------------+-----------+---------------------+---------------------+------------------------+

Features/parameters that are not mentioned here are universally supported for all 4 tree
methods (for instance, column sampling and constraints).  The `P` in external memory means
partially supported.  Please note that both categorical data and external memory are
experimental.
