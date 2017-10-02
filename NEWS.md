XGBoost Change Log
==================

This file records the changes in xgboost library in reverse chronological order.

## in progress version
* Updated Sklearn API
  - Updated to allow use of all XGBoost parameters via **kwargs.
  - Updated nthread to n_jobs and seed to random_state (as per Sklearn convention).
* Refactored gbm to allow more friendly cache strategy
  - Specialized some prediction routine
* Automatically remove nan from input data when it is sparse.
  - This can solve some of user reported problem of istart != hist.size
* Minor fixes
  - Thread local variable is upgraded so it is automatically freed at thread exit.
* Migrate to C++11
  - The current master version now requires C++11 enabled compiled(g++4.8 or higher)
* Predictor interface was factored out (in a manner similar to the updater interface).
* New functionality
  - Ability to adjust tree model's statistics to a new dataset without changing tree structures.
  - Extracting feature contributions to individual predictions.
  - Faster, histogram-based tree algorithm (`tree_method='hist'`) .
  - GPU/CUDA accelerated tree algorithms (`tree_method='gpu_hist'` or `'gpu_exact'`), including the GPU-based predictor.
* R package:
  - New parameters:
    - `silent` in `xgb.DMatrix()`
    - `use_int_id` in `xgb.model.dt.tree()`
    - `predcontrib` in `predict()`
  - Default value of the `save_period` parameter in `xgboost()` changed to NULL (consistent with `xgb.train()`).
  - It's possible to custom-build the R package with GPU acceleration support.

## v0.6 (2016.07.29)
* Version 0.5 is skipped due to major improvements in the core
* Major refactor of core library.
  - Goal: more flexible and modular code as a portable library.
  - Switch to use of c++11 standard code.
  - Random number generator defaults to ```std::mt19937```.
  - Share the data loading pipeline and logging module from dmlc-core.
  - Enable registry pattern to allow optionally plugin of objective, metric, tree constructor, data loader.
    - Future plugin modules can be put into xgboost/plugin and register back to the library.
  - Remove most of the raw pointers to smart ptrs, for RAII safety.
* Add official option to approximate algorithm `tree_method` to parameter.
  - Change default behavior to switch to prefer faster algorithm.
  - User will get a message when approximate algorithm is chosen.
* Change library name to libxgboost.so
* Backward compatiblity
  - The binary buffer file is not backward compatible with previous version.
  - The model file is backward compatible on 64 bit platforms.
* The model file is compatible between 64/32 bit platforms(not yet tested).
* External memory version and other advanced features will be exposed to R library as well on linux.
  - Previously some of the features are blocked due to C++11 and threading limits.
  - The windows version is still blocked due to Rtools do not support ```std::thread```.
* rabit and dmlc-core are maintained through git submodule
  - Anyone can open PR to update these dependencies now.
* Improvements
  - Rabit and xgboost libs are not thread-safe and use thread local PRNGs
  - This could fix some of the previous problem which runs xgboost on multiple threads.
* JVM Package
  - Enable xgboost4j for java and scala
  - XGBoost distributed now runs on Flink and Spark.
* Support model attributes listing for meta data.
  - https://github.com/dmlc/xgboost/pull/1198
  - https://github.com/dmlc/xgboost/pull/1166
* Support callback API
  - https://github.com/dmlc/xgboost/issues/892
  - https://github.com/dmlc/xgboost/pull/1211
  - https://github.com/dmlc/xgboost/pull/1264
* Support new booster DART(dropout in tree boosting)
  - https://github.com/dmlc/xgboost/pull/1220
* Add CMake build system
  - https://github.com/dmlc/xgboost/pull/1314

## v0.47 (2016.01.14)

* Changes in R library
  - fixed possible problem of poisson regression.
  - switched from 0 to NA for missing values.
  - exposed access to additional model parameters.
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

## v0.4 (2015.05.11)

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


## v0.3 (2014.09.07)

* Faster tree construction module
  - Allows subsample columns during tree construction via ```bst:col_samplebytree=ratio```
* Support for boosting from initial predictions
* Experimental version of LambdaRank
* Linear booster is now parallelized, using parallel coordinated descent.
* Add [Code Guide](src/README.md) for customizing objective function and evaluation
* Add R module


## v0.2x (2014.05.20)

* Python module
* Weighted samples instances
* Initial version of pairwise rank


## v0.1 (2014.03.26)

* Initial release
