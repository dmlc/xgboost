Change Log
=====

xgboost-0.1
=====
* Initial release

xgboost-0.2x
=====
* Python module
* Weighted samples instances
* Initial version of pairwise rank

xgboost-0.3
=====
* Faster tree construction module
  - Allows subsample columns during tree construction via ```bst:col_samplebytree=ratio```
* Support for boosting from initial predictions
* Experimental version of LambdaRank
* Linear booster is now parallelized, using parallel coordinated descent.
* Add [Code Guide](src/README.md) for customizing objective function and evaluation
* Add R module
