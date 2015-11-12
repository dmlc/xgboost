Change Log
==========

xgboost-0.1
===========
* Initial release

xgboost-0.2x
============
* Python module
* Weighted samples instances
* Initial version of pairwise rank

xgboost-0.3
===========
* Faster tree construction module
  - Allows subsample columns during tree construction via ```bst:col_samplebytree=ratio```
* Support for boosting from initial predictions
* Experimental version of LambdaRank
* Linear booster is now parallelized, using parallel coordinated descent.
* Add [Code Guide](src/README.md) for customizing objective function and evaluation
* Add R module

xgboost-0.4
===========
* Distributed version of xgboost that runs on YARN, scales to billions of examples
* Direct save/load data and model from/to S3 and HDFS
* Feature importance visualization in R module, by Michael Benesty
* Predict leaf index
* Poisson regression for counts data
* Early stopping option in training
* Native save load support in R and python
  - xgboost models now can be saved using save/load in R
  - xgboost python model is now pickable
* sklearn wrapper is supported in python module
* Experimental External memory version

on going at master
==================
* Changes in R library
  - fixed possible problem of poisson regression.
  - switched from 0 to NA for missing values.
* Changes in Python library
  - throws exception instead of crash terminal when a parameter error happens.
  - has importance plot and tree plot functions.
  - accepts different learning rates for each boosting round.
  - allows model training continuation from previously saved model.
  - allows early stopping in CV.
  - allows feval to return a list of tuples.
  - allows eval_metric to handle additional format.
  - improved compatibility in sklearn module.
  - additional parameters added for sklearn wrapper.
  - added pip installation functionality.
  - supports more Pandas DataFrame dtypes. 
  - added best_ntree_limit attribute, in addition to best_score and best_iteration.
* Java api is ready for use
* Added more test cases and continuous integration to make each build more robust.
