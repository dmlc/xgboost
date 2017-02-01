DART booster
============
[XGBoost](https://github.com/dmlc/xgboost)) mostly combines a huge number of regression trees with a small learning rate.
In this situation, trees added early are significant and trees added late are unimportant.

Rasmi et al. proposed a new method to add dropout techniques from the deep neural net community to boosted trees, and reported better results in some situations.

This is a instruction of new tree booster `dart`.

Original paper
--------------
Rashmi Korlakai Vinayak, Ran Gilad-Bachrach. "DART: Dropouts meet Multiple Additive Regression Trees." [JMLR](http://www.jmlr.org/proceedings/papers/v38/korlakaivinayak15.pdf)

Features
--------
- Drop trees in order to solve the over-fitting.
  - Trivial trees (to correct trivial errors) may be prevented.

Because of the randomness introduced in the training, expect the following few differences:
- Training can be slower than `gbtree` because the random dropout prevents usage of the prediction buffer.
- The early stop might not be stable, due to the randomness.

How it works
------------
- In ``$ m $``th training round, suppose ``$ k $`` trees are selected to be dropped.
- Let ``$ D = \sum_{i \in \mathbf{K}} F_i $`` be the leaf scores of dropped trees and ``$ F_m = \eta \tilde{F}_m $`` be the leaf scores of a new tree.
- The objective function is as follows:
```math
\mathrm{Obj}
= \sum_{j=1}^n L \left( y_j, \hat{y}_j^{m-1} - D_j + \tilde{F}_m \right)
+ \Omega \left( \tilde{F}_m \right).
```
- ``$ D $`` and ``$ F_m $`` are overshooting, so using scale factor
```math
\hat{y}_j^m = \sum_{i \not\in \mathbf{K}} F_i + a \left( \sum_{i \in \mathbf{K}} F_i + b F_m \right) .
```

Parameters
----------
### booster
* `dart`

This booster inherits `gbtree`, so `dart` has also `eta`, `gamma`, `max_depth` and so on.

Additional parameters are noted below.

### sample_type
type of sampling algorithm.
* `uniform`: (default) dropped trees are selected uniformly.
* `weighted`: dropped trees are selected in proportion to weight.

### normalize_type
type of normalization algorithm.
* `tree`: (default) New trees have the same weight of each of dropped trees.
```math
a \left( \sum_{i \in \mathbf{K}} F_i + \frac{1}{k} F_m \right)
&= a \left( \sum_{i \in \mathbf{K}} F_i + \frac{\eta}{k} \tilde{F}_m \right) \\
&\sim a \left( 1 + \frac{\eta}{k} \right) D \\
&= a \frac{k + \eta}{k} D = D , \\
&\quad a = \frac{k}{k + \eta} .
```

* `forest`: New trees have the same weight of sum of dropped trees (forest).
```math
a \left( \sum_{i \in \mathbf{K}} F_i + F_m \right)
&= a \left( \sum_{i \in \mathbf{K}} F_i + \eta \tilde{F}_m \right) \\
&\sim a \left( 1 + \eta \right) D \\
&= a (1 + \eta) D = D , \\
&\quad a = \frac{1}{1 + \eta} .
```

### rate_drop
dropout rate.
- range: [0.0, 1.0]

### skip_drop
probability of skipping dropout.
- If a dropout is skipped, new trees are added in the same manner as gbtree.
- range: [0.0, 1.0]

Sample Script
-------------
```python
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'booster': 'dart',
         'max_depth': 5, 'learning_rate': 0.1,
         'objective': 'binary:logistic', 'silent': True,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5}
num_round = 50
bst = xgb.train(param, dtrain, num_round)
# make prediction
# ntree_limit must not be 0
preds = bst.predict(dtest, ntree_limit=num_round)
```
