XGBoost Change Log
==================

This file records the changes in xgboost library in reverse chronological order.

## v0.7 (2017.12.30)
* **This version represents a major change from the last release (v0.6), which was released one year and half ago.**
* Updated Sklearn API
  - Add compatibility layer for scikit-learn v0.18: `sklearn.cross_validation` now deprecated
  - Updated to allow use of all XGBoost parameters via `**kwargs`.
  - Updated `nthread` to `n_jobs` and `seed` to `random_state` (as per Sklearn convention); `nthread` and `seed` are now marked as deprecated
  - Updated to allow choice of Booster (`gbtree`, `gblinear`, or `dart`)
  - `XGBRegressor` now supports instance weights (specify `sample_weight` parameter)
  - Pass `n_jobs` parameter to the `DMatrix` constructor
  - Add `xgb_model` parameter to `fit` method, to allow continuation of training
* Refactored gbm to allow more friendly cache strategy
  - Specialized some prediction routine
* Robust `DMatrix` construction from a sparse matrix
* Faster consturction of `DMatrix` from 2D NumPy matrices: elide copies, use of multiple threads
* Automatically remove nan from input data when it is sparse.
  - This can solve some of user reported problem of istart != hist.size
* Fix the single-instance prediction function to obtain correct predictions
* Minor fixes
  - Thread local variable is upgraded so it is automatically freed at thread exit.
  - Fix saving and loading `count::poisson` models
  - Fix CalcDCG to use base-2 logarithm
  - Messages are now written to stderr instead of stdout
  - Keep built-in evaluations while using customized evaluation functions
  - Use `bst_float` consistently to minimize type conversion
  - Copy the base margin when slicing `DMatrix`
  - Evaluation metrics are now saved to the model file
  - Use `int32_t` explicitly when serializing version
  - In distributed training, synchronize the number of features after loading a data matrix.
* Migrate to C++11
  - The current master version now requires C++11 enabled compiled(g++4.8 or higher)
* Predictor interface was factored out (in a manner similar to the updater interface).
* Makefile support for Solaris and ARM
* Test code coverage using Codecov
* Add CPP tests
* Add `Dockerfile` and `Jenkinsfile` to support continuous integration for GPU code
* New functionality
  - Ability to adjust tree model's statistics to a new dataset without changing tree structures.
  - Ability to extract feature contributions from individual predictions, as described in [here](http://blog.datadive.net/interpreting-random-forests/) and [here](https://arxiv.org/abs/1706.06060).
  - Faster, histogram-based tree algorithm (`tree_method='hist'`) .
  - GPU/CUDA accelerated tree algorithms (`tree_method='gpu_hist'` or `'gpu_exact'`), including the GPU-based predictor.
  - Monotonic constraints: when other features are fixed, force the prediction to be monotonic increasing with respect to a certain specified feature.
  - Faster gradient caculation using AVX SIMD
  - Ability to export models in JSON format
  - Support for Tweedie regression
  - Additional dropout options for DART: binomial+1, epsilon
  - Ability to update an existing model in-place: this is useful for many applications, such as determining feature importance
* Python package:
  - New parameters:
    - `learning_rates` in `cv()`
    - `shuffle` in `mknfold()`
    - `max_features` and `show_values` in `plot_importance()`
    - `sample_weight` in `XGBRegressor.fit()`
  - Support binary wheel builds
  - Fix `MultiIndex` detection to support Pandas 0.21.0 and higher
  - Support metrics and evaluation sets whose names contain `-`
  - Support feature maps when plotting trees
  - Compatibility fix for Python 2.6
  - Call `print_evaluation` callback at last iteration
  - Use appropriate integer types when calling native code, to prevent truncation and memory error
  - Fix shared library loading on Mac OS X 
* R package:
  - New parameters:
    - `silent` in `xgb.DMatrix()`
    - `use_int_id` in `xgb.model.dt.tree()`
    - `predcontrib` in `predict()`
    - `monotone_constraints` in `xgb.train()`
  - Default value of the `save_period` parameter in `xgboost()` changed to NULL (consistent with `xgb.train()`).
  - It's possible to custom-build the R package with GPU acceleration support.
  - Enable JVM build for Mac OS X and Windows
  - Integration with AppVeyor CI
  - Improved safety for garbage collection
  - Store numeric attributes with higher precision
  - Easier installation for devel version
  - Improved `xgb.plot.tree()`
  - Various minor fixes to improve user experience and robustness
  - Register native code to pass CRAN check
  - Updated CRAN submission
* JVM packages
  - Add Spark pipeline persistence API
  - Fix data persistence: loss evaluation on test data had wrongly used caches for training data.
  - Clean external cache after training
  - Implement early stopping
  - Enable training of multiple models by distinguishing stage IDs
  - Better Spark integration: support RDD / dataframe / dataset, integrate with Spark ML package
  - XGBoost4j now supports ranking task
  - Support training with missing data
  - Refactor JVM package to separate regression and classification models to be consistent with other machine learning libraries
  - Support XGBoost4j compilation on Windows
  - Parameter tuning tool
  - Publish source code for XGBoost4j to maven local repo
  - Scala implementation of the Rabit tracker (drop-in replacement for the Java implementation)
  - Better exception handling for the Rabit tracker
  - Persist `num_class`, number of classes (for classification task)
  - `XGBoostModel` now holds `BoosterParams`
  - libxgboost4j is now part of CMake build
  - Release `DMatrix` when no longer needed, to conserve memory
  - Expose `baseMargin`, to allow initialization of boosting with predictions from an external model
  - Support instance weights
  - Use `SparkParallelismTracker` to prevent jobs from hanging forever
  - Expose train-time evaluation metrics via `XGBoostModel.summary`
  - Option to specify `host-ip` explicitly in the Rabit tracker 
* Documentation
  - Better math notation for gradient boosting
  - Updated build instructions for Mac OS X
  - Template for GitHub issues
  - Add `CITATION` file for citing XGBoost in scientific writing
  - Fix dropdown menu in xgboost.readthedocs.io
  - Document `updater_seq` parameter
  - Style fixes for Python documentation
  - Links to additional examples and tutorials
  - Clarify installation requirements
* Changes that break backward compatibility
  - [#1519](https://github.com/dmlc/xgboost/pull/1519) XGBoost-spark no longer contains APIs for DMatrix; use the public booster interface instead.
  - [#2476](https://github.com/dmlc/xgboost/pull/2476) `XGBoostModel.predict()` now has a different signature


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
