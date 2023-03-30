XGBoost Change Log
==================

This file records the changes in xgboost library in reverse chronological order.

## 1.7.5 (2023 Mar 30)
This is a patch release for bug fixes.

* C++ requirement is updated to C++-17, along with which, CUDA 11.8 is used as the default CTK. (#8860, #8855, #8853)
* Fix import for pyspark ranker. (#8692)
* Fix Windows binary wheel to be compatible with Poetry (#8991)
* Fix GPU hist with column sampling. (#8850)
* Make sure iterative DMatrix is properly initialized. (#8997)
* [R] Update link in document. (#8998)

## 1.7.4 (2023 Feb 16)
This is a patch release for bug fixes.

* [R] Fix OpenMP detection on macOS. (#8684)
* [Python] Make sure input numpy array is aligned. (#8690)
* Fix feature interaction with column sampling in gpu_hist evaluator. (#8754)
* Fix GPU L1 error. (#8749)
* [PySpark] Fix feature types param (#8772)
* Fix ranking with quantile dmatrix and group weight. (#8762)

## 1.7.3 (2023 Jan 6)
This is a patch release for bug fixes.

* [Breaking] XGBoost Sklearn estimator method `get_params` no longer returns internally configured values. (#8634)
* Fix linalg iterator, which may crash the L1 error. (#8603)
* Fix loading pickled GPU model with a CPU-only XGBoost build. (#8632)
* Fix inference with unseen categories with categorical features. (#8591, #8602)
* CI fixes. (#8620, #8631, #8579)

## v1.7.2 (2022 Dec 8)
This is a patch release for bug fixes.

* Work with newer thrust and libcudacxx (#8432)
* Support null value in CUDA array interface namespace. (#8486)
* Use `getsockname` instead of `SO_DOMAIN` on AIX. (#8437)
* [pyspark] Make QDM optional based on a cuDF check (#8471)
* [pyspark] sort qid for SparkRanker. (#8497)
* [dask] Properly await async method client.wait_for_workers. (#8558)

* [R] Fix CRAN test notes. (#8428)

* [doc] Fix outdated document [skip ci]. (#8527)
* [CI] Fix github action mismatched glibcxx. (#8551)

## v1.7.1 (2022 Nov 3)
This is a patch release to incorporate the following hotfix:

* Add back xgboost.rabit for backwards compatibility (#8411)


## v1.7.0 (2022 Oct 20)

We are excited to announce the feature packed XGBoost 1.7 release. The release note will walk through some of the major new features first, then make a summary for other improvements and language-binding-specific changes.

### PySpark

XGBoost 1.7 features initial support for PySpark integration. The new interface is adapted from the existing PySpark XGBoost interface developed by databricks with additional features like `QuantileDMatrix` and the rapidsai plugin (GPU pipeline) support. The new Spark XGBoost Python estimators not only benefit from PySpark ml facilities for powerful distributed computing but also enjoy the rest of the Python ecosystem. Users can define a custom objective, callbacks, and metrics in Python and use them with this interface on distributed clusters. The support is labeled as experimental with more features to come in future releases. For a brief introduction please visit the tutorial on XGBoost's [document page](https://xgboost.readthedocs.io/en/latest/tutorials/spark_estimator.html). (#8355, #8344, #8335, #8284, #8271, #8283, #8250, #8231, #8219, #8245, #8217, #8200, #8173, #8172, #8145, #8117, #8131, #8088, #8082, #8085, #8066, #8068, #8067, #8020, #8385)

Due to its initial support status, the new interface has some limitations; categorical features and multi-output models are not yet supported.

### Development of categorical data support
More progress on the experimental support for categorical features. In 1.7, XGBoost can handle missing values in categorical features and features a new parameter `max_cat_threshold`, which limits the number of categories that can be used in the split evaluation. The parameter is enabled when the partitioning algorithm is used and helps prevent over-fitting. Also, the sklearn interface can now accept the `feature_types` parameter to use data types other than dataframe for categorical features. (#8280, #7821, #8285, #8080, #7948, #7858, #7853, #8212, #7957, #7937, #7934)


###  Experimental support for federated learning and new communication collective

An exciting addition to XGBoost is the experimental federated learning support. The federated learning is implemented with a gRPC federated server that aggregates allreduce calls, and federated clients that train on local data and use existing tree methods (approx, hist, gpu_hist). Currently, this only supports horizontal federated learning (samples are split across participants, and each participant has all the features and labels). Future plans include vertical federated learning (features split across participants), and stronger privacy guarantees with homomorphic encryption and differential privacy. See [Demo with NVFlare integration](demo/nvflare/README.md) for example usage with nvflare.

As part of the work, XGBoost 1.7 has replaced the old rabit module with the new collective module as the network communication interface with added support for runtime backend selection. In previous versions, the backend is defined at compile time and can not be changed once built. In this new release, users can choose between `rabit` and `federated.` (#8029, #8351, #8350, #8342, #8340, #8325, #8279, #8181, #8027, #7958, #7831, #7879, #8257, #8316, #8242, #8057, #8203, #8038, #7965, #7930, #7911)

The feature is available in the public PyPI binary package for testing.

### Quantile DMatrix
Before 1.7, XGBoost has an internal data structure called `DeviceQuantileDMatrix` (and its distributed version). We now extend its support to CPU and renamed it to `QuantileDMatrix`. This data structure is used for optimizing memory usage for the `hist` and `gpu_hist` tree methods. The new feature helps reduce CPU memory usage significantly, especially for dense data. The new `QuantileDMatrix` can be initialized from both CPU and GPU data, and regardless of where the data comes from, the constructed instance can be used by both the CPU algorithm and GPU algorithm including training and prediction (with some overhead of conversion if the device of data and training algorithm doesn't match). Also, a new parameter `ref` is added to `QuantileDMatrix`, which can be used to construct validation/test datasets. Lastly, it's set as default in the scikit-learn interface when a supported tree method is specified by users. (#7889, #7923, #8136, #8215, #8284, #8268, #8220, #8346, #8327, #8130, #8116, #8103, #8094, #8086, #7898, #8060, #8019, #8045, #7901, #7912, #7922)

### Mean absolute error
The mean absolute error is a new member of the collection of objectives in XGBoost. It's noteworthy since MAE has zero hessian value, which is unusual to XGBoost as XGBoost relies on Newton optimization. Without valid Hessian values, the convergence speed can be slow. As part of the support for MAE, we added line searches into the XGBoost training algorithm to overcome the difficulty of training without valid Hessian values. In the future, we will extend the line search to other objectives where it's appropriate for faster convergence speed. (#8343, #8107, #7812, #8380)

### XGBoost on Browser
With the help of the [pyodide](https://github.com/pyodide/pyodide) project, you can now run XGBoost on browsers. (#7954, #8369)

### Experimental IPv6 Support for Dask

With the growing adaption of the new internet protocol, XGBoost joined the club. In the latest release, the Dask interface can be used on IPv6 clusters, see XGBoost's Dask tutorial for details. (#8225, #8234)

### Optimizations
We have new optimizations for both the `hist` and `gpu_hist` tree methods to make XGBoost's training even more efficient.

* Hist
Hist now supports optional by-column histogram build, which is automatically configured based on various conditions of input data. This helps the XGBoost CPU hist algorithm to scale better with different shapes of training datasets. (#8233, #8259). Also, the build histogram kernel now can better utilize CPU registers (#8218)

* GPU Hist
GPU hist performance is significantly improved for wide datasets. GPU hist now supports batched node build, which reduces kernel latency and increases throughput. The improvement is particularly significant when growing deep trees with the default ``depthwise`` policy. (#7919, #8073, #8051, #8118, #7867, #7964, #8026)

### Breaking Changes
Breaking changes made in the 1.7 release are summarized below.
- The  `grow_local_histmaker`  updater is removed. This updater is rarely used in practice and has no test. We decided to remove it and focus have XGBoot focus on other more efficient algorithms. (#7992, #8091)
- Single precision histogram is removed due to its lack of accuracy caused by significant floating point error. In some cases the error can be difficult to detect due to log-scale operations, which makes the parameter dangerous to use. (#7892, #7828)
- Deprecated CUDA architectures are no longer supported in the release binaries. (#7774)
- As part of the federated learning development, the `rabit` module is replaced with the new `collective` module. It's a drop-in replacement with added runtime backend selection, see the federated learning section for more details (#8257)

### General new features and improvements
Before diving into package-specific changes, some general new features other than those listed at the beginning are summarized here.
* Users of `DMatrix` and `QuantileDMatrix` can get the data from XGBoost. In previous versions, only getters for meta info like labels are available. The new method is available in Python (`DMatrix::get_data`) and C. (#8269, #8323)
* In previous versions, the GPU histogram tree method may generate phantom gradient for missing values due to floating point error. We fixed such an error in this release and XGBoost is much better equated to handle floating point errors when training on GPU. (#8274, #8246)
* Parameter validation is no longer experimental. (#8206)
* C pointer parameters and JSON parameters are vigorously checked. (#8254, #8254)
* Improved handling of JSON model input. (#7953, #7918)
* Support IBM i OS (#7920, #8178)

### Fixes
Some noteworthy bug fixes that are not related to specific language binding are listed in this section.
* Rename misspelled config parameter for pseudo-Huber (#7904)
* Fix feature weights with nested column sampling. (#8100)
* Fix loading DMatrix binary in distributed env. (#8149)
* Force auc.cc to be statically linked for unusual compiler platforms. (#8039)
* New logic for detecting libomp on macos (#8384).

### Python Package
* Python 3.8 is now the minimum required Python version. (#8071)
* More progress on type hint support. Except for the new PySpark interface, the XGBoost module is fully typed. (#7742, #7945, #8302, #7914, #8052)
* XGBoost now validates the feature names in `inplace_predict`, which also affects the predict function in scikit-learn estimators as it uses `inplace_predict` internally. (#8359)
* Users can now get the data from `DMatrix` using `DMatrix::get_data` or `QuantileDMatrix::get_data`.
* Show `libxgboost.so` path in build info. (#7893)
* Raise import error when using the sklearn module while scikit-learn is missing. (#8049)
* Use `config_context` in the sklearn interface. (#8141)
* Validate features for inplace prediction. (#8359)
* Pandas dataframe handling is refactored to reduce data fragmentation. (#7843)
* Support more pandas nullable types (#8262)
* Remove pyarrow workaround. (#7884)

* Binary wheel size
We aim to enable as many features as possible in XGBoost's default binary distribution on PyPI (package installed with pip), but there's a upper limit on the size of the binary wheel. In 1.7, XGBoost reduces the size of the wheel by pruning unused CUDA architectures. (#8179, #8152, #8150)

* Fixes
  Some noteworthy fixes are listed here:
  - Fix the Dask interface with the latest cupy. (#8210)
  - Check cuDF lazily to avoid potential errors with cuda-python. (#8084)
* Fix potential error in DMatrix constructor on 32-bit platform. (#8369)

* Maintenance work
  - Linter script is moved from dmlc-core to XGBoost with added support for formatting, mypy, and parallel run, along with some fixes (#7967, #8101, #8216)
  - We now require the use of `isort` and `black` for selected files. (#8137, #8096)
  - Code cleanups. (#7827)
  - Deprecate `use_label_encoder` in XGBClassifier. The label encoder has already been deprecated and removed in the previous version. These changes only affect the indicator parameter (#7822)
  - Remove the use of distutils. (#7770)
  - Refactor and fixes for tests (#8077, #8064, #8078, #8076, #8013, #8010, #8244, #7833)

* Documents
  - [dask] Fix potential error in demo. (#8079)
  - Improved documentation for the ranker. (#8356, #8347)
  - Indicate lack of py-xgboost-gpu on Windows (#8127)
  - Clarification for feature importance. (#8151)
  - Simplify Python getting started example (#8153)

### R Package
We summarize improvements for the R package briefly here:
* Feature info including names and types are now passed to DMatrix in preparation for categorical feature support. (#804)
* XGBoost 1.7 can now gracefully load old R models from RDS for better compatibility with 3-party tuning libraries (#7864)
* The R package now can be built with parallel compilation, along with fixes for warnings in CRAN tests. (#8330)
* Emit error early if DiagrammeR is missing (#8037)
* Fix R package Windows build. (#8065)

### JVM Packages
The consistency between JVM packages and other language bindings is greatly improved in 1.7, improvements range from model serialization format to the default value of hyper-parameters.

* Java package now supports feature names and feature types for DMatrix in preparation for categorical feature support. (#7966)
* Models trained by the JVM packages can now be safely used with other language bindings. (#7896, #7907)
* Users can specify the model format when saving models with a stream. (#7940, #7955)
* The default value for training parameters is now sourced from XGBoost directly, which helps JVM packages be consistent with other packages. (#7938)
* Set the correct objective if the user doesn't explicitly set it (#7781)
* Auto-detection of MUSL is replaced by system properties (#7921)
* Improved error message for launching tracker. (#7952, #7968)
* Fix a race condition in parameter configuration. (#8025)
* [Breaking] ` timeoutRequestWorkers` is now removed. With the support for barrier mode, this parameter is no longer needed. (#7839)
* Dependencies updates. (#7791, #8157, #7801, #8240)

### Documents
- Document for the C interface is greatly improved and is now displayed at the [sphinx document page](https://xgboost.readthedocs.io/en/latest/c.html). Thanks to the breathe project, you can view the C API just like the Python API. (#8300)
- We now avoid having XGBoost internal text parser in demos and recommend users use dedicated libraries for loading data whenever it's feasible. (#7753)
- Python survival training demos are now displayed at [sphinx gallery](https://xgboost.readthedocs.io/en/latest/python/survival-examples/index.html). (#8328)
- Some typos, links, format, and grammar fixes. (#7800, #7832, #7861, #8099, #8163, #8166, #8229, #8028, #8214, #7777, #7905, #8270, #8309, d70e59fef, #7806)
- Updated winning solution under readme.md (#7862)
- New security policy. (#8360)
- GPU document is overhauled as we consider CUDA support to be feature-complete. (#8378)

### Maintenance
* Code refactoring and cleanups. (#7850, #7826, #7910, #8332, #8204)
* Reduce compiler warnings. (#7768, #7916, #8046, #8059, #7974, #8031, #8022)
* Compiler workarounds. (#8211, #8314, #8226, #8093)
* Dependencies update. (#8001, #7876, #7973, #8298, #7816)
* Remove warnings emitted in previous versions. (#7815)
* Small fixes occurred during development. (#8008)

### CI and Tests
* We overhauled the CI infrastructure to reduce the CI cost and lift the maintenance burdens. Jenkins is replaced with buildkite for better automation, with which, finer control of test runs is implemented to reduce overall cost. Also, we refactored some of the existing tests to reduce their runtime, drooped the size of docker images, and removed multi-GPU C++ tests. Lastly, `pytest-timeout` is added as an optional dependency for running Python tests to keep the test time in check. (#7772, #8291, #8286, #8276, #8306, #8287, #8243, #8313, #8235, #8288, #8303, #8142, #8092, #8333, #8312, #8348)
* New documents for how to reproduce the CI environment (#7971, #8297)
* Improved automation for JVM release. (#7882)
* GitHub Action security-related updates. (#8263, #8267, #8360)
* Other fixes and maintenance work. (#8154, #7848, #8069, #7943)
* Small updates and fixes to GitHub action pipelines. (#8364, #8321, #8241, #7950, #8011)

## v1.6.1 (2022 May 9)
This is a patch release for bug fixes and Spark barrier mode support. The R package is unchanged.

### Experimental support for categorical data
- Fix segfault when the number of samples is smaller than the number of categories. (https://github.com/dmlc/xgboost/pull/7853)
- Enable partition-based split for all model types. (https://github.com/dmlc/xgboost/pull/7857)

### JVM packages
We replaced the old parallelism tracker with spark barrier mode to improve the robustness of the JVM package and fix the GPU training pipeline.
- Fix GPU training pipeline quantile synchronization. (#7823, #7834)
- Use barrier model in spark package. (https://github.com/dmlc/xgboost/pull/7836, https://github.com/dmlc/xgboost/pull/7840, https://github.com/dmlc/xgboost/pull/7845, https://github.com/dmlc/xgboost/pull/7846)
- Fix shared object loading on some platforms. (https://github.com/dmlc/xgboost/pull/7844)

## v1.6.0 (2022 Apr 16)

After a long period of development, XGBoost v1.6.0 is packed with many new features and
improvements. We summarize them in the following sections starting with an introduction to
some major new features, then moving on to language binding specific changes including new
features and notable bug fixes for that binding.

### Development of categorical data support
This version of XGBoost features new improvements and full coverage of experimental
categorical data support in Python and C package with tree model.  Both `hist`, `approx`
and `gpu_hist` now support training with categorical data.  Also, partition-based
categorical split is introduced in this release. This split type is first available in
LightGBM in the context of gradient boosting. The previous XGBoost release supported one-hot split where the splitting criteria is of form `x \in {c}`, i.e. the categorical feature `x` is tested against a single candidate. The new release allows for more expressive conditions: `x \in S` where the categorical feature `x` is tested against multiple candidates. Moreover, it is now possible to use any tree algorithms (`hist`, `approx`, `gpu_hist`) when creating categorical splits. For more
information, please see our tutorial on [categorical
data](https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html), along with
examples linked on that page. (#7380, #7708, #7695, #7330, #7307, #7322, #7705,
#7652, #7592, #7666, #7576, #7569, #7529, #7575, #7393, #7465, #7385, #7371, #7745, #7810)

In the future, we will continue to improve categorical data support with new features and
optimizations. Also, we are looking forward to bringing the feature beyond Python binding,
contributions and feedback are welcomed! Lastly, as a result of experimental status, the
behavior might be subject to change, especially the default value of related
hyper-parameters.

### Experimental support for multi-output model

XGBoost 1.6 features initial support for the multi-output model, which includes
multi-output regression and multi-label classification. Along with this, the XGBoost
classifier has proper support for base margin without to need for the user to flatten the
input. In this initial support, XGBoost builds one model for each target similar to the
sklearn meta estimator, for more details, please see our [quick
introduction](https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html).

(#7365, #7736, #7607, #7574, #7521, #7514, #7456, #7453, #7455, #7434, #7429, #7405, #7381)

### External memory support
External memory support for both approx and hist tree method is considered feature
complete in XGBoost 1.6.  Building upon the iterator-based interface introduced in the
previous version, now both `hist` and `approx` iterates over each batch of data during
training and prediction.  In previous versions, `hist` concatenates all the batches into
an internal representation, which is removed in this version.  As a result, users can
expect higher scalability in terms of data size but might experience lower performance due
to disk IO. (#7531, #7320, #7638, #7372)

### Rewritten approx

The `approx` tree method is rewritten based on the existing `hist` tree method. The
rewrite closes the feature gap between `approx` and `hist` and improves the performance.
Now the behavior of `approx` should be more aligned with `hist` and `gpu_hist`. Here is a
list of user-visible changes:

- Supports both `max_leaves` and `max_depth`.
- Supports `grow_policy`.
- Supports monotonic constraint.
- Supports feature weights.
- Use `max_bin` to replace `sketch_eps`.
- Supports categorical data.
- Faster performance for many of the datasets.
- Improved performance and robustness for distributed training.
- Supports prediction cache.
- Significantly better performance for external memory when `depthwise` policy is used.

### New serialization format
Based on the existing JSON serialization format, we introduce UBJSON support as a more
efficient alternative. Both formats will be available in the future and we plan to
gradually [phase out](https://github.com/dmlc/xgboost/issues/7547) support for the old
binary model format.  Users can opt to use the different formats in the serialization
function by providing the file extension `json` or `ubj`. Also, the `save_raw` function in
all supported languages bindings gains a new parameter for exporting the model in different
formats, available options are `json`, `ubj`, and `deprecated`, see document for the
language binding you are using for details. Lastly, the default internal serialization
format is set to UBJSON, which affects Python pickle and R RDS. (#7572, #7570, #7358,
#7571, #7556, #7549, #7416)

### General new features and improvements
Aside from the major new features mentioned above, some others are summarized here:

* Users can now access the build information of XGBoost binary in Python and C
  interface. (#7399, #7553)
* Auto-configuration of `seed_per_iteration` is removed, now distributed training should
  generate closer results to single node training when sampling is used. (#7009)
* A new parameter `huber_slope` is introduced for the `Pseudo-Huber` objective.
* During source build, XGBoost can choose cub in the system path automatically. (#7579)
* XGBoost now honors the CPU counts from CFS, which is usually set in docker
  environments. (#7654, #7704)
* The metric `aucpr` is rewritten for better performance and GPU support. (#7297, #7368)
* Metric calculation is now performed in double precision. (#7364)
* XGBoost no longer mutates the global OpenMP thread limit. (#7537, #7519, #7608, #7590,
  #7589, #7588, #7687)
* The default behavior of `max_leave` and `max_depth` is now unified (#7302, #7551).
* CUDA fat binary is now compressed. (#7601)
* Deterministic result for evaluation metric and linear model. In previous versions of
  XGBoost, evaluation results might differ slightly for each run due to parallel reduction
  for floating-point values, which is now addressed. (#7362, #7303, #7316, #7349)
* XGBoost now uses double for GPU Hist node sum, which improves the accuracy of
  `gpu_hist`. (#7507)

### Performance improvements
Most of the performance improvements are integrated into other refactors during feature
developments. The `approx` should see significant performance gain for many datasets as
mentioned in the previous section, while the `hist` tree method also enjoys improved
performance with the removal of the internal `pruner` along with some other
refactoring. Lastly, `gpu_hist` no longer synchronizes the device during training. (#7737)

### General bug fixes
This section lists bug fixes that are not specific to any language binding.
* The `num_parallel_tree` is now a model parameter instead of a training hyper-parameter,
  which fixes model IO with random forest. (#7751)
* Fixes in CMake script for exporting configuration. (#7730)
* XGBoost can now handle unsorted sparse input. This includes text file formats like
  libsvm and scipy sparse matrix where column index might not be sorted. (#7731)
* Fix tree param feature type, this affects inputs with the number of columns greater than
  the maximum value of int32. (#7565)
* Fix external memory with gpu_hist and subsampling. (#7481)
* Check the number of trees in inplace predict, this avoids a potential segfault when an
  incorrect value for `iteration_range` is provided. (#7409)
* Fix non-stable result in cox regression (#7756)

### Changes in the Python package
Other than the changes in Dask, the XGBoost Python package gained some new features and
improvements along with small bug fixes.

* Python 3.7 is required as the lowest Python version. (#7682)
* Pre-built binary wheel for Apple Silicon. (#7621, #7612, #7747) Apple Silicon users will
  now be able to run `pip install xgboost` to install XGBoost.
* MacOS users no longer need to install `libomp` from Homebrew, as the XGBoost wheel now
  bundles `libomp.dylib` library.
* There are new parameters for users to specify the custom metric with new
  behavior. XGBoost can now output transformed prediction values when a custom objective is
  not supplied.  See our explanation in the
  [tutorial](https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html#reverse-link-function)
  for details.
* For the sklearn interface, following the estimator guideline from scikit-learn, all
  parameters in `fit` that are not related to input data are moved into the constructor
  and can be set by `set_params`. (#6751, #7420, #7375, #7369)
* Apache arrow format is now supported, which can bring better performance to users'
  pipeline (#7512)
* Pandas nullable types are now supported (#7760)
* A new function `get_group` is introduced for `DMatrix` to allow users to get the group
  information in the custom objective function. (#7564)
* More training parameters are exposed in the sklearn interface instead of relying on the
  `**kwargs`. (#7629)
* A new attribute `feature_names_in_` is defined for all sklearn estimators like
  `XGBRegressor` to follow the convention of sklearn. (#7526)
* More work on Python type hint. (#7432, #7348, #7338, #7513, #7707)
* Support the latest pandas Index type. (#7595)
* Fix for Feature shape mismatch error on s390x platform (#7715)
* Fix using feature names for constraints with multiple groups (#7711)
* We clarified the behavior of the callback function when it contains mutable
  states. (#7685)
* Lastly, there are some code cleanups and maintenance work. (#7585, #7426, #7634, #7665,
  #7667, #7377, #7360, #7498, #7438, #7667, #7752, #7749, #7751)

### Changes in the Dask interface
* Dask module now supports user-supplied host IP and port address of scheduler node.
  Please see [introduction](https://xgboost.readthedocs.io/en/latest/tutorials/dask.html#troubleshooting) and
  [API document](https://xgboost.readthedocs.io/en/latest/python/python_api.html#optional-dask-configuration)
  for reference. (#7645, #7581)
* Internal `DMatrix` construction in dask now honers thread configuration. (#7337)
* A fix for `nthread` configuration using the Dask sklearn interface. (#7633)
* The Dask interface can now handle empty partitions.  An empty partition is different
  from an empty worker, the latter refers to the case when a worker has no partition of an
  input dataset, while the former refers to some partitions on a worker that has zero
  sizes. (#7644, #7510)
* Scipy sparse matrix is supported as Dask array partition. (#7457)
* Dask interface is no longer considered experimental. (#7509)

### Changes in the R package
This section summarizes the new features, improvements, and bug fixes to the R package.

* `load.raw` can optionally construct a booster as return. (#7686)
* Fix parsing decision stump, which affects both transforming text representation to data
  table and plotting. (#7689)
* Implement feature weights. (#7660)
* Some improvements for complying the CRAN release policy. (#7672, #7661, #7763)
* Support CSR data for predictions (#7615)
* Document update (#7263, #7606)
* New maintainer for the CRAN package (#7691, #7649)
* Handle non-standard installation of toolchain on macos (#7759)

### Changes in JVM-packages
Some new features for JVM-packages are introduced for a more integrated GPU pipeline and
better compatibility with musl-based Linux. Aside from this, we have a few notable bug
fixes.

* User can specify the tracker IP address for training, which helps running XGBoost on
  restricted network environments. (#7808)
* Add support for detecting musl-based Linux (#7624)
* Add `DeviceQuantileDMatrix` to Scala binding (#7459)
* Add Rapids plugin support, now more of the JVM pipeline can be accelerated by RAPIDS (#7491, #7779, #7793, #7806)
* The setters for CPU and GPU are more aligned (#7692, #7798)
* Control logging for early stopping (#7326)
* Do not repartition when nWorker = 1 (#7676)
* Fix the prediction issue for `multi:softmax` (#7694)
* Fix for serialization of custom objective and eval (#7274)
* Update documentation about Python tracker (#7396)
* Remove jackson from dependency, which fixes CVE-2020-36518. (#7791)
* Some refactoring to the training pipeline for better compatibility between CPU and
  GPU. (#7440, #7401, #7789, #7784)
* Maintenance work. (#7550, #7335, #7641, #7523, #6792, #4676)

### Deprecation
Other than the changes in the Python package and serialization, we removed some deprecated
features in previous releases. Also, as mentioned in the previous section, we plan to
phase out the old binary format in future releases.

* Remove old warning in 1.3 (#7279)
* Remove label encoder deprecated in 1.3. (#7357)
* Remove old callback deprecated in 1.3. (#7280)
* Pre-built binary will no longer support deprecated CUDA architectures including sm35 and
  sm50. Users can continue to use these platforms with source build. (#7767)

### Documentation
This section lists some of the general changes to XGBoost's document, for language binding
specific change please visit related sections.

* Document is overhauled to use the new RTD theme, along with integration of Python
  examples using Sphinx gallery. Also, we replaced most of the hard-coded URLs with sphinx
  references. (#7347, #7346, #7468, #7522, #7530)
* Small update along with fixes for broken links, typos, etc. (#7684, #7324, #7334, #7655,
  #7628, #7623, #7487, #7532, #7500, #7341, #7648, #7311)
* Update document for GPU. [skip ci] (#7403)
* Document the status of RTD hosting. (#7353)
* Update document for building from source. (#7664)
* Add note about CRAN release [skip ci] (#7395)

### Maintenance
This is a summary of maintenance work that is not specific to any language binding.

* Add CMake option to use /MD runtime (#7277)
* Add clang-format configuration. (#7383)
* Code cleanups (#7539, #7536, #7466, #7499, #7533, #7735, #7722, #7668, #7304, #7293,
  #7321, #7356, #7345, #7387, #7577, #7548, #7469, #7680, #7433, #7398)
* Improved tests with better coverage and latest dependency (#7573, #7446, #7650, #7520,
  #7373, #7723, #7611, #7771)
* Improved automation of the release process. (#7278, #7332, #7470)
* Compiler workarounds (#7673)
* Change shebang used in CLI demo. (#7389)
* Update affiliation (#7289)

### CI
Some fixes and update to XGBoost's CI infrastructure. (#7739, #7701, #7382, #7662, #7646,
#7582, #7407, #7417, #7475, #7474, #7479, #7472, #7626)


## v1.5.0 (2021 Oct 11)

This release comes with many exciting new features and optimizations, along with some bug
fixes.  We will describe the experimental categorical data support and the external memory
interface independently. Package-specific new features will be listed in respective
sections.

### Development on categorical data support
In version 1.3, XGBoost introduced an experimental feature for handling categorical data
natively, without one-hot encoding. XGBoost can fit categorical splits in decision
trees. (Currently, the generated splits will be of form `x \in {v}`, where the input is
compared to a single category value. A future version of XGBoost will generate splits that
compare the input against a list of multiple category values.)

Most of the other features, including prediction, SHAP value computation, feature
importance, and model plotting were revised to natively handle categorical splits.  Also,
all Python interfaces including native interface with and without quantized `DMatrix`,
scikit-learn interface, and Dask interface now accept categorical data with a wide range
of data structures support including numpy/cupy array and cuDF/pandas/modin dataframe.  In
practice, the following are required for enabling categorical data support during
training:

  - Use Python package.
  - Use `gpu_hist` to train the model.
  - Use JSON model file format for saving the model.

Once the model is trained, it can be used with most of the features that are available on
the Python package.  For a quick introduction, see
https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html

Related PRs: (#7011, #7001, #7042, #7041, #7047, #7043, #7036, #7054, #7053, #7065, #7213, #7228, #7220, #7221, #7231, #7306)

* Next steps

	- Revise the CPU training algorithm to handle categorical data natively and generate categorical splits
	- Extend the CPU and GPU algorithms to generate categorical splits of form `x \in S`
	where the input is compared with multiple category values.  split. (#7081)

### External memory
This release features a brand-new interface and implementation for external memory (also
known as out-of-core training).  (#6901, #7064, #7088, #7089, #7087, #7092, #7070,
#7216). The new implementation leverages the data iterator interface, which is currently
used to create `DeviceQuantileDMatrix`. For a quick introduction, see
https://xgboost.readthedocs.io/en/latest/tutorials/external_memory.html#data-iterator
. During the development of this new interface, `lz4` compression is removed. (#7076).
Please note that external memory support is still experimental and not ready for
production use yet.  All future development will focus on this new interface and users are
advised to migrate. (You are using the old interface if you are using a URL suffix to use
external memory.)

### New features in Python package
* Support numpy array interface and all numeric types from numpy in `DMatrix`
  construction and `inplace_predict` (#6998, #7003).  Now XGBoost no longer makes data
  copy when input is numpy array view.
* The early stopping callback in Python has a new `min_delta` parameter to control the
  stopping behavior (#7137)
* Python package now supports calculating feature scores for the linear model, which is
  also available on R package. (#7048)
* Python interface now supports configuring constraints using feature names instead of
  feature indices.
* Typehint support for more Python code including scikit-learn interface and rabit
  module. (#6799, #7240)
* Add tutorial for XGBoost-Ray (#6884)

### New features in R package
* In 1.4 we have a new prediction function in the C API which is used by the Python
  package.  This release revises the R package to use the new prediction function as well.
  A new parameter `iteration_range` for the predict function is available, which can be
  used for specifying the range of trees for running prediction. (#6819, #7126)
* R package now supports the `nthread` parameter in `DMatrix` construction. (#7127)

### New features in JVM packages
* Support GPU dataframe and `DeviceQuantileDMatrix` (#7195).  Constructing `DMatrix`
  with GPU data structures and the interface for quantized `DMatrix` were first
  introduced in the Python package and are now available in the xgboost4j package.
* JVM packages now support saving and getting early stopping attributes. (#7095) Here is a
  quick [example](https://github.com/dmlc/xgboost/jvm-packages/xgboost4j-example/src/main/java/ml/dmlc/xgboost4j/java/example/EarlyStopping.java "example") in JAVA (#7252).

### General new features
* We now have a pre-built binary package for R on Windows with GPU support. (#7185)
* CUDA compute capability 86 is now part of the default CMake build configuration with
  newly added support for CUDA 11.4. (#7131, #7182, #7254)
* XGBoost can be compiled using system CUB provided by CUDA 11.x installation. (#7232)

### Optimizations
The performance for both `hist` and `gpu_hist` has been significantly improved in 1.5
with the following optimizations:
* GPU multi-class model training now supports prediction cache. (#6860)
* GPU histogram building is sped up and the overall training time is 2-3 times faster on
  large datasets (#7180, #7198).  In addition, we removed the parameter `deterministic_histogram` and now
  the GPU algorithm is always deterministic.
* CPU hist has an optimized procedure for data sampling (#6922)
* More performance optimization in regression and binary classification objectives on
  CPU (#7206)
* Tree model dump is now performed in parallel (#7040)

### Breaking changes
* `n_gpus` was deprecated in 1.0 release and is now removed.
* Feature grouping in CPU hist tree method is removed, which was disabled long
  ago. (#7018)
* C API for Quantile DMatrix is changed to be consistent with the new external memory
  implementation. (#7082)

### Notable general bug fixes
* XGBoost no long changes global CUDA device ordinal when `gpu_id` is specified (#6891,
  #6987)
* Fix `gamma` negative likelihood evaluation metric. (#7275)
* Fix integer value of `verbose_eal` for `xgboost.cv` function in Python. (#7291)
* Remove extra sync in CPU hist for dense data, which can lead to incorrect tree node
  statistics. (#7120, #7128)
* Fix a bug in GPU hist when data size is larger than `UINT32_MAX` with missing
  values. (#7026)
* Fix a thread safety issue in prediction with the `softmax` objective. (#7104)
* Fix a thread safety issue in CPU SHAP value computation. (#7050) Please note that all
  prediction functions in Python are thread-safe.
* Fix model slicing. (#7149, #7078)
* Workaround a bug in old GCC which can lead to segfault during construction of
  DMatrix. (#7161)
* Fix histogram truncation in GPU hist, which can lead to slightly-off results. (#7181)
* Fix loading GPU linear model pickle files on CPU-only machine. (#7154)
* Check input value is duplicated when CPU quantile queue is full (#7091)
* Fix parameter loading with training continuation. (#7121)
* Fix CMake interface for exposing C library by specifying dependencies. (#7099)
* Callback and early stopping are explicitly disabled for the scikit-learn interface
  random forest estimator. (#7236)
* Fix compilation error on x86 (32-bit machine) (#6964)
* Fix CPU memory usage with extremely sparse datasets (#7255)
* Fix a bug in GPU multi-class AUC implementation with weighted data (#7300)

### Python package
Other than the items mentioned in the previous sections, there are some Python-specific
improvements.
* Change development release postfix to `dev` (#6988)
* Fix early stopping behavior with MAPE metric (#7061)
* Fixed incorrect feature mismatch error message (#6949)
* Add predictor to skl constructor. (#7000, #7159)
* Re-enable feature validation in predict proba. (#7177)
* scikit learn interface regression estimator now can pass the scikit-learn estimator
  check and is fully compatible with scikit-learn utilities.  `__sklearn_is_fitted__` is
  implemented as part of the changes (#7130, #7230)
* Conform the latest pylint. (#7071, #7241)
* Support latest panda range index in DMatrix construction. (#7074)
* Fix DMatrix construction from pandas series. (#7243)
* Fix typo and grammatical mistake in error message (#7134)
* [dask] disable work stealing explicitly for training tasks (#6794)
* [dask] Set dataframe index in predict. (#6944)
* [dask] Fix prediction on df with latest dask. (#6969)
* [dask] Fix dask predict on `DaskDMatrix` with `iteration_range`. (#7005)
* [dask] Disallow importing non-dask estimators from xgboost.dask (#7133)

### R package
Improvements other than new features on R package:
* Optimization for updating R handles in-place (#6903)
* Removed the magrittr dependency. (#6855, #6906, #6928)
* The R package now hides all C++ symbols to avoid conflicts. (#7245)
* Other maintenance including code cleanups, document updates. (#6863, #6915, #6930, #6966, #6967)

### JVM packages
Improvements other than new features on JVM packages:
* Constructors with implicit missing value are deprecated due to confusing behaviors. (#7225)
* Reduce scala-compiler, scalatest dependency scopes (#6730)
* Making the Java library loader emit helpful error messages on missing dependencies. (#6926)
* JVM packages now use the Python tracker in XGBoost instead of dmlc.  The one in XGBoost
  is shared between JVM packages and Python Dask and enjoys better maintenance (#7132)
* Fix "key not found: train" error (#6842)
* Fix model loading from stream (#7067)

### General document improvements
* Overhaul the installation documents. (#6877)
* A few demos are added for AFT with dask (#6853), callback with dask (#6995), inference
  in C (#7151), `process_type`. (#7135)
* Fix PDF format of document. (#7143)
* Clarify the behavior of `use_rmm`. (#6808)
* Clarify prediction function. (#6813)
* Improve tutorial on feature interactions (#7219)
* Add small example for dask sklearn interface. (#6970)
* Update Python intro.  (#7235)
* Some fixes/updates (#6810, #6856, #6935, #6948, #6976, #7084, #7097, #7170, #7173, #7174, #7226, #6979, #6809, #6796, #6979)

### Maintenance
* Some refactoring around CPU hist, which lead to better performance but are listed under general maintenance tasks:
  - Extract evaluate splits from CPU hist. (#7079)
  - Merge lossgude and depthwise strategies for CPU hist (#7007)
  - Simplify sparse and dense CPU hist kernels (#7029)
  - Extract histogram builder from CPU Hist. (#7152)

* Others
  - Fix `gpu_id` with custom objective. (#7015)
  - Fix typos in AUC. (#6795)
  - Use constexpr in `dh::CopyIf`. (#6828)
  - Update dmlc-core. (#6862)
  - Bump version to 1.5.0 snapshot in master. (#6875)
  - Relax shotgun test. (#6900)
  - Guard against index error in prediction. (#6982)
  - Hide symbols in CI build + hide symbols for C and CUDA (#6798)
  - Persist data in dask test. (#7077)
  - Fix typo in arguments of PartitionBuilder::Init (#7113)
  - Fix typo in src/common/hist.cc BuildHistKernel (#7116)
  - Use upstream URI in distributed quantile tests. (#7129)
  - Include cpack (#7160)
  - Remove synchronization in monitor. (#7164)
  - Remove unused code. (#7175)
  - Fix building on CUDA 11.0. (#7187)
  - Better error message for `ncclUnhandledCudaError`. (#7190)
  - Add noexcept to JSON objects. (#7205)
  - Improve wording for warning (#7248)
  - Fix typo in release script. [skip ci] (#7238)
  - Relax shotgun test. (#6918)
  - Relax test for decision stump in distributed environment. (#6919)
  -	[dask] speed up tests (#7020)

### CI
* [CI] Rotate access keys for uploading MacOS artifacts from Travis CI (#7253)
* Reduce Travis environment setup time. (#6912)
* Restore R cache on github action. (#6985)
* [CI] Remove stray build artifact to avoid error in artifact packaging (#6994)
* [CI] Move appveyor tests to action (#6986)
* Remove appveyor badge. [skip ci] (#7035)
* [CI] Configure RAPIDS, dask, modin (#7033)
* Test on s390x. (#7038)
* [CI] Upgrade to CMake 3.14 (#7060)
* [CI] Update R cache. (#7102)
* [CI] Pin libomp to 11.1.0  (#7107)
* [CI] Upgrade build image to CentOS 7 + GCC 8; require CUDA 10.1 and later (#7141)
* [dask] Work around segfault in prediction. (#7112)
* [dask] Remove the workaround for segfault. (#7146)
* [CI] Fix hanging Python setup in Windows CI (#7186)
* [CI] Clean up in beginning of each task in Win CI (#7189)
* Fix travis. (#7237)

### Acknowledgement
* **Contributors**: Adam Pocock (@Craigacp), Jeff H (@JeffHCross), Johan Hansson (@JohanWork), Jose Manuel Llorens (@JoseLlorensRipolles), Benjamin Szőke (@Livius90), @ReeceGoding, @ShvetsKS, Robert Zabel (@ZabelTech), Ali (@ali5h), Andrew Ziem (@az0), Andy Adinets (@canonizer), @david-cortes, Daniel Saxton (@dsaxton), Emil Sadek (@esadek), @farfarawayzyt, Gil Forsyth (@gforsyth), @giladmaya, @graue70, Philip Hyunsu Cho (@hcho3), James Lamb (@jameslamb), José Morales (@jmoralez), Kai Fricke (@krfricke), Christian Lorentzen (@lorentzenchr), Mads R. B. Kristensen (@madsbk), Anton Kostin (@masguit42), Martin Petříček (@mpetricek-corp), @naveenkb, Taewoo Kim (@oOTWK), Viktor Szathmáry (@phraktle), Robert Maynard (@robertmaynard), TP Boudreau (@tpboudreau), Jiaming Yuan (@trivialfis), Paul Taylor (@trxcllnt), @vslaykovsky, Bobby Wang (@wbo4958),
* **Reviewers**: Nan Zhu (@CodingCat), Adam Pocock (@Craigacp), Jose Manuel Llorens (@JoseLlorensRipolles), Kodi Arfer (@Kodiologist), Benjamin Szőke (@Livius90), Mark Guryanov (@MarkGuryanov), Rory Mitchell (@RAMitchell), @ReeceGoding, @ShvetsKS, Egor Smirnov (@SmirnovEgorRu), Andrew Ziem (@az0), @candalfigomoro, Andy Adinets (@canonizer), Dante Gama Dessavre (@dantegd), @david-cortes, Daniel Saxton (@dsaxton), @farfarawayzyt, Gil Forsyth (@gforsyth), Harutaka Kawamura (@harupy), Philip Hyunsu Cho (@hcho3), @jakirkham, James Lamb (@jameslamb), José Morales (@jmoralez), James Bourbeau (@jrbourbeau), Christian Lorentzen (@lorentzenchr), Martin Petříček (@mpetricek-corp), Nikolay Petrov (@napetrov), @naveenkb, Viktor Szathmáry (@phraktle), Robin Teuwens (@rteuwens), Yuan Tang (@terrytangyuan), TP Boudreau (@tpboudreau), Jiaming Yuan (@trivialfis), @vkuzmin-uber, Bobby Wang (@wbo4958), William Hicks (@wphicks)


## v1.4.2 (2021.05.13)
This is a patch release for Python package with following fixes:

* Handle the latest version of cupy.ndarray in inplace_predict. (#6933)
* Ensure output array from predict_leaf is (n_samples, ) when there's only 1 tree. 1.4.0 outputs (n_samples, 1). (#6889)
* Fix empty dataset handling with multi-class AUC. (#6947)
* Handle object type from pandas in inplace_predict. (#6927)


## v1.4.1 (2021.04.20)
This is a bug fix release.

* Fix GPU implementation of AUC on some large datasets. (#6866)

## v1.4.0 (2021.04.12)

### Introduction of pre-built binary package for R, with GPU support
Starting with release 1.4.0, users now have the option of installing `{xgboost}` without
having to build it from the source. This is particularly advantageous for users who want
to take advantage of the GPU algorithm (`gpu_hist`), as previously they'd have to build
`{xgboost}` from the source using CMake and NVCC. Now installing `{xgboost}` with GPU
support is as easy as: `R CMD INSTALL ./xgboost_r_gpu_linux.tar.gz`. (#6827)

See the instructions at https://xgboost.readthedocs.io/en/latest/build.html

### Improvements on prediction functions
XGBoost has many prediction types including shap value computation and inplace prediction.
In 1.4 we overhauled the underlying prediction functions for C API and Python API with an
unified interface. (#6777, #6693, #6653, #6662, #6648, #6668, #6804)
* Starting with 1.4, sklearn interface prediction will use inplace predict by default when
  input data is supported.
* Users can use inplace predict with `dart` booster and enable GPU acceleration just
  like `gbtree`.
* Also all prediction functions with tree models are now thread-safe.  Inplace predict is
  improved with `base_margin` support.
* A new set of C predict functions are exposed in the public interface.
* A user-visible change is a newly added parameter called `strict_shape`.  See
  https://xgboost.readthedocs.io/en/latest/prediction.html for more details.


### Improvement on Dask interface
* Starting with 1.4, the Dask interface is considered to be feature-complete, which means
  all of the models found in the single node Python interface are now supported in Dask,
  including but not limited to ranking and random forest.  Also, the prediction function
  is significantly faster and supports shap value computation.
  - Most of the parameters found in single node sklearn interface are supported by
    Dask interface. (#6471, #6591)
  - Implements learning to rank.  On the Dask interface, we use the newly added support of
    query ID to enable group structure. (#6576)
  - The Dask interface has Python type hints support. (#6519)
  - All models can be safely pickled. (#6651)
  - Random forest estimators are now supported. (#6602)
  - Shap value computation is now supported. (#6575, #6645, #6614)
  - Evaluation result is printed on the scheduler process. (#6609)
  - `DaskDMatrix` (and device quantile dmatrix) now accepts all meta-information. (#6601)

* Prediction optimization.  We enhanced and speeded up the prediction function for the
  Dask interface.  See the latest Dask tutorial page in our document for an overview of
  how you can optimize it even further. (#6650, #6645, #6648, #6668)

* Bug fixes
  - If you are using the latest Dask and distributed where `distributed.MultiLock` is
    present, XGBoost supports training multiple models on the same cluster in
    parallel. (#6743)
  - A bug fix for when using `dask.client` to launch async task, XGBoost might use a
    different client object internally. (#6722)

* Other improvements on documents, blogs, tutorials, and demos. (#6389, #6366, #6687,
  #6699, #6532, #6501)

### Python package
With changes from Dask and general improvement on prediction, we have made some
enhancements on the general Python interface and IO for booster information.  Starting
from 1.4, booster feature names and types can be saved into the JSON model.  Also some
model attributes like `best_iteration`, `best_score` are restored upon model load.  On
sklearn interface, some attributes are now implemented as Python object property with
better documents.

* Breaking change: All `data` parameters in prediction functions are renamed to `X`
  for better compliance to sklearn estimator interface guidelines.
* Breaking change: XGBoost used to generate some pseudo feature names with `DMatrix`
  when inputs like `np.ndarray` don't have column names.  The procedure is removed to
  avoid conflict with other inputs. (#6605)
* Early stopping with training continuation is now supported. (#6506)
* Optional import for Dask and cuDF are now lazy. (#6522)
* As mentioned in the prediction improvement summary, the sklearn interface uses inplace
  prediction whenever possible. (#6718)
* Booster information like feature names and feature types are now saved into the JSON
  model file. (#6605)
* All `DMatrix` interfaces including `DeviceQuantileDMatrix` and counterparts in Dask
  interface (as mentioned in the Dask changes summary) now accept all the meta-information
  like `group` and `qid` in their constructor for better consistency. (#6601)
* Booster attributes are restored upon model load so users don't have to call `attr`
  manually. (#6593)
* On sklearn interface, all models accept `base_margin` for evaluation datasets. (#6591)
* Improvements over the setup script including smaller sdist size and faster installation
  if the C++ library is already built (#6611, #6694, #6565).

* Bug fixes for Python package:
  - Don't validate feature when number of rows is 0. (#6472)
  - Move metric configuration into booster. (#6504)
  - Calling XGBModel.fit() should clear the Booster by default (#6562)
  - Support `_estimator_type`. (#6582)
  - [dask, sklearn] Fix predict proba. (#6566, #6817)
  - Restore unknown data support. (#6595)
  - Fix learning rate scheduler with cv. (#6720)
  - Fixes small typo in sklearn documentation (#6717)
  - [python-package] Fix class Booster: feature_types = None (#6705)
  - Fix divide by 0 in feature importance when no split is found. (#6676)


### JVM package
* [jvm-packages] fix early stopping doesn't work even without custom_eval setting (#6738)
* fix potential TaskFailedListener's callback won't be called (#6612)
* [jvm] Add ability to load booster direct from byte array (#6655)
* [jvm-packages] JVM library loader extensions (#6630)

### R package
* R documentation: Make construction of DMatrix consistent.
* Fix R documentation for xgb.train. (#6764)

### ROC-AUC
We re-implemented the ROC-AUC metric in XGBoost.  The new implementation supports
multi-class classification and has better support for learning to rank tasks that are not
binary.  Also, it has a better-defined average on distributed environments with additional
handling for invalid datasets. (#6749, #6747, #6797)

### Global configuration.
Starting from 1.4, XGBoost's Python, R and C interfaces support a new global configuration
model where users can specify some global parameters.  Currently, supported parameters are
`verbosity` and `use_rmm`.  The latter is experimental, see rmm plugin demo and
related README file for details. (#6414, #6656)

### Other New features.
* Better handling for input data types that support `__array_interface__`.  For some
  data types including GPU inputs and `scipy.sparse.csr_matrix`, XGBoost employs
  `__array_interface__` for processing the underlying data.  Starting from 1.4, XGBoost
  can accept arbitrary array strides (which means column-major is supported) without
  making data copies, potentially reducing a significant amount of memory consumption.
  Also version 3 of `__cuda_array_interface__` is now supported.  (#6776, #6765, #6459,
  #6675)
* Improved parameter validation, now feeding XGBoost with parameters that contain
  whitespace will trigger an error. (#6769)
* For Python and R packages, file paths containing the home indicator `~` are supported.
* As mentioned in the Python changes summary, the JSON model can now save feature
  information of the trained booster.  The JSON schema is updated accordingly. (#6605)
* Development of categorical data support is continued.  Newly added weighted data support
  and `dart` booster support. (#6508, #6693)
* As mentioned in Dask change summary, ranking now supports the `qid` parameter for
  query groups. (#6576)
* `DMatrix.slice` can now consume a numpy array. (#6368)

### Other breaking changes
* Aside from the feature name generation, there are 2 breaking changes:
  - Drop saving binary format for memory snapshot. (#6513, #6640)
  - Change default evaluation metric for binary:logitraw objective to logloss (#6647)

### CPU Optimization
* Aside from the general changes on predict function, some optimizations are applied on
  CPU implementation. (#6683, #6550, #6696, #6700)
* Also performance for sampling initialization in `hist` is improved. (#6410)

### Notable fixes in the core library
These fixes do not reside in particular language bindings:
* Fixes for gamma regression.  This includes checking for invalid input values, fixes for
  gamma deviance metric, and better floating point guard for gamma negative log-likelihood
  metric. (#6778, #6537, #6761)
* Random forest with `gpu_hist` might generate low accuracy in previous versions. (#6755)
* Fix a bug in GPU sketching when data size exceeds limit of 32-bit integer. (#6826)
* Memory consumption fix for row-major adapters (#6779)
* Don't estimate sketch batch size when rmm is used. (#6807) (#6830)
* Fix in-place predict with missing value. (#6787)
* Re-introduce double buffer in UpdatePosition, to fix perf regression in gpu_hist (#6757)
* Pass correct split_type to GPU predictor (#6491)
* Fix DMatrix feature names/types IO. (#6507)
* Use view for `SparsePage` exclusively to avoid some data access races. (#6590)
* Check for invalid data. (#6742)
* Fix relocatable include in CMakeList (#6734) (#6737)
* Fix DMatrix slice with feature types. (#6689)

### Other deprecation notices:

* This release will be the last release to support CUDA 10.0. (#6642)

* Starting in the next release, the Python package will require Pip 19.3+ due to the use
  of manylinux2014 tag. Also, CentOS 6, RHEL 6 and other old distributions will not be
  supported.

### Known issue:

MacOS build of the JVM packages doesn't support multi-threading out of the box. To enable
multi-threading with JVM packages, MacOS users will need to build the JVM packages from
the source. See https://xgboost.readthedocs.io/en/latest/jvm/index.html#installation-from-source


### Doc
* Dedicated page for `tree_method` parameter is added. (#6564, #6633)
* [doc] Add FLAML as a fast tuning tool for XGBoost  (#6770)
* Add document for tests directory. [skip ci] (#6760)
* Fix doc string of config.py to use correct `versionadded` (#6458)
* Update demo for prediction. (#6789)
* [Doc] Document that AUCPR is for binary classification/ranking (#5899)
* Update the C API comments (#6457)
* Fix document. [skip ci] (#6669)

### Maintenance: Testing, continuous integration
* Use CPU input for test_boost_from_prediction. (#6818)
* [CI] Upload xgboost4j.dll to S3 (#6781)
* Update dmlc-core submodule (#6745)
* [CI] Use manylinux2010_x86_64 container to vendor libgomp (#6485)
* Add conda-forge badge (#6502)
* Fix merge conflict. (#6512)
* [CI] Split up main.yml, add mypy. (#6515)
* [Breaking] Upgrade cuDF and RMM to 0.18 nightlies; require RMM 0.18+ for RMM plugin (#6510)
* "featue_map" typo changed to  "feature_map" (#6540)
* Add script for generating release tarball. (#6544)
* Add credentials to .gitignore (#6559)
* Remove warnings in tests. (#6554)
* Update dmlc-core submodule and conform to new API (#6431)
* Suppress hypothesis health check for dask client. (#6589)
* Fix pylint. (#6714)
* [CI] Clear R package cache (#6746)
* Exclude dmlc test on github action. (#6625)
* Tests for regression metrics with weights. (#6729)
* Add helper script and doc for releasing pip package. (#6613)
* Support pylint 2.7.0 (#6726)
* Remove R cache in github action. (#6695)
* [CI] Do not mix up stashed executable built for ARM and x86_64 platforms (#6646)
* [CI] Add ARM64 test to Jenkins pipeline (#6643)
* Disable s390x and arm64 tests on travis for now. (#6641)
* Move sdist test to action. (#6635)
* [dask] Rework base margin test. (#6627)


### Maintenance: Refactor code for legibility and maintainability
* Improve OpenMP exception handling (#6680)
* Improve string view to reduce string allocation. (#6644)
* Simplify Span checks. (#6685)
* Use generic dispatching routine for array interface. (#6672)


## v1.3.0 (2020.12.08)

### XGBoost4J-Spark: Exceptions should cancel jobs gracefully instead of killing SparkContext (#6019).
* By default, exceptions in XGBoost4J-Spark causes the whole SparkContext to shut down, necessitating the restart of the Spark cluster. This behavior is often a major inconvenience.
* Starting from 1.3.0 release, XGBoost adds a new parameter `killSparkContextOnWorkerFailure` to optionally prevent killing SparkContext. If this parameter is set, exceptions will gracefully cancel training jobs instead of killing SparkContext.

### GPUTreeSHAP: GPU acceleration of the TreeSHAP algorithm (#6038, #6064, #6087, #6099, #6163, #6281, #6332)
* [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) is a game theoretic approach to explain predictions of machine learning models. It computes feature importance scores for individual examples, establishing how each feature influences a particular prediction. TreeSHAP is an optimized SHAP algorithm specifically designed for decision tree ensembles.
* Starting with 1.3.0 release, it is now possible to leverage CUDA-capable GPUs to accelerate the TreeSHAP algorithm. Check out [the demo notebook](https://github.com/dmlc/xgboost/blob/master/demo/gpu_acceleration/shap.ipynb).
* The CUDA implementation of the TreeSHAP algorithm is hosted at [rapidsai/GPUTreeSHAP](https://github.com/rapidsai/gputreeshap). XGBoost imports it as a Git submodule.

### New style Python callback API (#6199, #6270, #6320, #6348, #6376, #6399, #6441)
* The XGBoost Python package now offers a re-designed callback API. The new callback API lets you design various extensions of training in idomatic Python. In addition, the new callback API allows you to use early stopping with the native Dask API (`xgboost.dask`). Check out [the tutorial](https://xgboost.readthedocs.io/en/release_1.3.0/python/callbacks.html) and [the demo](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/callbacks.py).

### Enable the use of `DeviceQuantileDMatrix` / `DaskDeviceQuantileDMatrix` with large data (#6201, #6229, #6234).
* `DeviceQuantileDMatrix` can achieve memory saving by avoiding extra copies of the training data, and the saving is bigger for large data. Unfortunately, large data with more than 2^31 elements was triggering integer overflow bugs in CUB and Thrust. Tracking issue: #6228.
* This release contains a series of work-arounds to allow the use of `DeviceQuantileDMatrix` with large data:
  - Loop over `copy_if` (#6201)
  - Loop over `thrust::reduce` (#6229)
  - Implement the inclusive scan algorithm in-house, to handle large offsets (#6234)

### Support slicing of tree models (#6302)
* Accessing the best iteration of a model after the application of early stopping used to be error-prone, need to manually pass the `ntree_limit` argument to the `predict()` function.
* Now we provide a simple interface to slice tree models by specifying a range of boosting rounds. The tree ensemble can be split into multiple sub-ensembles via the slicing interface. Check out [an example](https://xgboost.readthedocs.io/en/release_1.3.0/python/model.html).
* In addition, the early stopping callback now supports `save_best` option. When enabled, XGBoost will save (persist) the model at the best boosting round and discard the trees that were fit subsequent to the best round.

### Weighted subsampling of features (columns) (#5962)
* It is now possible to sample features (columns) via weighted subsampling, in which features with higher weights are more likely to be selected in the sample. Weighted subsampling allows you to encode domain knowledge by emphasizing a particular set of features in the choice of tree splits. In addition, you can prevent particular features from being used in any splits, by assigning them zero weights.
* Check out [the demo](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/feature_weights.py).

### Improved integration with Dask
* Support reverse-proxy environment such as Google Kubernetes Engine (#6343, #6475)
* An XGBoost training job will no longer use all available workers. Instead, it will only use the workers that contain input data (#6343).
* The new callback API works well with the Dask training API.
* The `predict()` and `fit()` function of `DaskXGBClassifier` and `DaskXGBRegressor` now accept a base margin (#6155).
* Support more meta data in the Dask API (#6130, #6132, #6333).
* Allow passing extra keyword arguments as `kwargs` in `predict()` (#6117)
* Fix typo in dask interface: `sample_weights` -> `sample_weight` (#6240)
* Allow empty data matrix in AFT survival, as Dask may produce empty partitions (#6379)
* Speed up prediction by overlapping prediction jobs in all workers (#6412)

### Experimental support for direct splits with categorical features (#6028, #6128, #6137, #6140, #6164, #6165, #6166, #6179, #6194, #6219)
* Currently, XGBoost requires users to one-hot-encode categorical variables. This has adverse performance implications, as the creation of many dummy variables results into higher memory consumption and may require fitting deeper trees to achieve equivalent model accuracy.
* The 1.3.0 release of XGBoost contains an experimental support for direct handling of categorical variables in test nodes. Each test node will have the condition of form `feature_value \in match_set`, where the `match_set` on the right hand side contains one or more matching categories. The matching categories in `match_set` represent the condition for traversing to the right child node. Currently, XGBoost will only generate categorical splits with only a single matching category ("one-vs-rest split"). In a future release, we plan to remove this restriction and produce splits with multiple matching categories in `match_set`.
* The categorical split requires the use of JSON model serialization. The legacy binary serialization method cannot be used to save (persist) models with categorical splits.
* Note. This feature is currently highly experimental. Use it at your own risk. See the detailed list of limitations at [#5949](https://github.com/dmlc/xgboost/pull/5949).

### Experimental plugin for RAPIDS Memory Manager (#5873, #6131, #6146, #6150, #6182)
* RAPIDS Memory Manager library ([rapidsai/rmm](https://github.com/rapidsai/rmm)) provides a collection of efficient memory allocators for NVIDIA GPUs. It is now possible to use XGBoost with memory allocators provided by RMM, by enabling the RMM integration plugin. With this plugin, XGBoost is now able to share a common GPU memory pool with other applications using RMM, such as the RAPIDS data science packages.
* See [the demo](https://github.com/dmlc/xgboost/blob/master/demo/rmm_plugin/README.md) for a working example, as well as directions for building XGBoost with the RMM plugin.
* The plugin will be soon considered non-experimental, once #6297 is resolved.

### Experimental plugin for oneAPI programming model (#5825)
* oneAPI is a programming interface developed by Intel aimed at providing one programming model for many types of hardware such as CPU, GPU, FGPA and other hardware accelerators.
* XGBoost now includes an experimental plugin for using oneAPI for the predictor and objective functions. The plugin is hosted in the directory `plugin/updater_oneapi`.
* Roadmap: #5442

### Pickling the XGBoost model will now trigger JSON serialization (#6027)
* The pickle will now contain the JSON string representation of the XGBoost model, as well as related configuration.

### Performance improvements
* Various performance improvement on multi-core CPUs
  - Optimize DMatrix build time by up to 3.7x. (#5877)
  - CPU predict performance improvement, by up to 3.6x. (#6127)
  - Optimize CPU sketch allreduce for sparse data (#6009)
  - Thread local memory allocation for BuildHist, leading to speedup up to 1.7x. (#6358)
  - Disable hyperthreading for DMatrix creation (#6386). This speeds up DMatrix creation by up to 2x.
  - Simple fix for static shedule in predict (#6357)
* Unify thread configuration, to make it easy to utilize all CPU cores (#6186)
* [jvm-packages] Clean the way deterministic paritioning is computed (#6033)
* Speed up JSON serialization by implementing an intrusive pointer class (#6129). It leads to 1.5x-2x performance boost.

### API additions
* [R] Add SHAP summary plot using ggplot2 (#5882)
* Modin DataFrame can now be used as input (#6055)
* [jvm-packages] Add `getNumFeature` method (#6075)
* Add MAPE metric (#6119)
* Implement GPU predict leaf. (#6187)
* Enable cuDF/cuPy inputs in `XGBClassifier` (#6269)
* Document tree method for feature weights. (#6312)
* Add `fail_on_invalid_gpu_id` parameter, which will cause XGBoost to terminate upon seeing an invalid value of `gpu_id` (#6342)

### Breaking: the default evaluation metric for classification is changed to `logloss` / `mlogloss` (#6183)
* The default metric used to be accuracy, and it is not statistically consistent to perform early stopping with the accuracy metric when we are really optimizing the log loss for the `binary:logistic` objective.
* For statistical consistency, the default metric for classification has been changed to `logloss`. Users may choose to preserve the old behavior by explicitly specifying `eval_metric`.

### Breaking: `skmaker` is now removed (#5971)
* The `skmaker` updater has not been documented nor tested.

### Breaking: the JSON model format no longer stores the leaf child count (#6094).
* The leaf child count field has been deprecated and is not used anywhere in the XGBoost codebase.

### Breaking: XGBoost now requires MacOS 10.14 (Mojave) and later.
* Homebrew has dropped support for MacOS 10.13 (High Sierra), so we are not able to install the OpenMP runtime (`libomp`) from Homebrew on MacOS 10.13. Please use MacOS 10.14 (Mojave) or later.

### Deprecation notices
* The use of `LabelEncoder` in `XGBClassifier` is now deprecated and will be removed in the next minor release (#6269). The deprecation is necessary to support multiple types of inputs, such as cuDF data frames or cuPy arrays.
* The use of certain positional arguments in the Python interface is deprecated (#6365). Users will use deprecation warnings for the use of position arguments for certain function parameters. New code should use keyword arguments as much as possible. We have not yet decided when we will fully require the use of keyword arguments.

### Bug-fixes
* On big-endian arch, swap the byte order in the binary serializer to enable loading models that were produced by a little-endian machine (#5813).
* [jvm-packages] Fix deterministic partitioning with dataset containing Double.NaN (#5996)
* Limit tree depth for GPU hist to 31 to prevent integer overflow (#6045)
* [jvm-packages] Set `maxBins` to 256 to align with the default value in the C++ code (#6066)
* [R] Fix CRAN check (#6077)
* Add back support for `scipy.sparse.coo_matrix` (#6162)
* Handle duplicated values in sketching. (#6178)
* Catch all standard exceptions in C API. (#6220)
* Fix linear GPU input (#6255)
* Fix inplace prediction interval. (#6259)
* [R] allow `xgb.plot.importance()` calls to fill a grid (#6294)
* Lazy import dask libraries. (#6309)
* Deterministic data partitioning for external memory (#6317)
* Avoid resetting seed for every configuration. (#6349)
* Fix label errors in graph visualization (#6369)
* [jvm-packages] fix potential unit test suites aborted issue due to race condition (#6373)
* [R] Fix warnings from `R check --as-cran` (#6374)
* [R] Fix a crash that occurs with noLD R (#6378)
* [R] Do not convert continuous labels to factors (#6380)
* [R] remove uses of `exists()` (#6387)
* Propagate parameters to the underlying `Booster` handle from `XGBClassifier.set_param` / `XGBRegressor.set_param`. (#6416)
* [R] Fix R package installation via CMake (#6423)
* Enforce row-major order in cuPy array (#6459)
* Fix filtering callable objects in the parameters passed to the scikit-learn API. (#6466)

### Maintenance: Testing, continuous integration, build system
* [CI] Improve JVM test in GitHub Actions (#5930)
* Refactor plotting test so that it can run independently (#6040)
* [CI] Cancel builds on subsequent pushes (#6011)
* Fix Dask Pytest fixture (#6024)
* [CI] Migrate linters to GitHub Actions (#6035)
* [CI] Remove win2016 JVM test from GitHub Actions (#6042)
* Fix CMake build with `BUILD_STATIC_LIB` option (#6090)
* Don't link imported target in CMake (#6093)
* Work around a compiler bug in MacOS AppleClang 11 (#6103)
* [CI] Fix CTest by running it in a correct directory (#6104)
* [R] Check warnings explicitly for model compatibility tests (#6114)
* [jvm-packages] add xgboost4j-gpu/xgboost4j-spark-gpu module to facilitate release (#6136)
* [CI] Time GPU tests. (#6141)
* [R] remove warning in configure.ac (#6152)
* [CI] Upgrade cuDF and RMM to 0.16 nightlies; upgrade to Ubuntu 18.04 (#6157)
* [CI] Test C API demo (#6159)
* Option for generating device debug info. (#6168)
* Update `.gitignore` (#6175, #6193, #6346)
* Hide C++ symbols from dmlc-core (#6188)
* [CI] Added arm64 job in Travis-CI (#6200)
* [CI] Fix Docker build for CUDA 11 (#6202)
* [CI] Move non-OpenMP gtest to GitHub Actions (#6210)
* [jvm-packages] Fix up build for xgboost4j-gpu, xgboost4j-spark-gpu (#6216)
* Add more tests for categorical data support (#6219)
* [dask] Test for data initializaton. (#6226)
* Bump junit from 4.11 to 4.13.1 in /jvm-packages/xgboost4j (#6230)
* Bump junit from 4.11 to 4.13.1 in /jvm-packages/xgboost4j-gpu (#6233)
* [CI] Reduce testing load with RMM (#6249)
* [CI] Build a Python wheel for aarch64 platform (#6253)
* [CI] Time the CPU tests on Jenkins. (#6257)
* [CI] Skip Dask tests on ARM. (#6267)
* Fix a typo in `is_arm()` in testing.py (#6271)
* [CI] replace `egrep` with `grep -E` (#6287)
* Support unity build. (#6295)
* [CI] Mark flaky tests as XFAIL (#6299)
* [CI] Use separate Docker cache for each CUDA version (#6305)
* Added `USE_NCCL_LIB_PATH` option to enable user to set `NCCL_LIBRARY` during build  (#6310)
* Fix flaky data initialization test. (#6318)
* Add a badge for GitHub Actions (#6321)
* Optional `find_package` for sanitizers. (#6329)
* Use pytest conventions consistently in Python tests (#6337)
* Fix missing space in warning message (#6340)
* Update `custom_metric_obj.rst` (#6367)
* [CI] Run R check with `--as-cran` flag on GitHub Actions (#6371)
* [CI] Remove R check from Jenkins (#6372)
* Mark GPU external memory test as XFAIL. (#6381)
* [CI] Add noLD R test (#6382)
* Fix MPI build. (#6403)
* [CI] Upgrade to MacOS Mojave image (#6406)
* Fix flaky sparse page dmatrix test. (#6417)
* [CI] Upgrade cuDF and RMM to 0.17 nightlies (#6434)
* [CI] Fix CentOS 6 Docker images (#6467)
* [CI] Vendor libgomp in the manylinux Python wheel (#6461)
* [CI] Hot fix for libgomp vendoring (#6482)

### Maintenance: Clean up and merge the Rabit submodule (#6023, #6095, #6096, #6105, #6110, #6262, #6275, #6290)
* The Rabit submodule is now maintained as part of the XGBoost codebase.
* Tests for Rabit are now part of the test suites of XGBoost.
* Rabit can now be built on the Windows platform.
* We made various code re-formatting for the C++ code with clang-tidy.
* Public headers of XGBoost no longer depend on Rabit headers.
* Unused CMake targets for Rabit were removed.
* Single-point model recovery has been dropped and removed from Rabit, simplifying the Rabit code greatly. The single-point model recovery feature has not been adequately maintained over the years.
* We removed the parts of Rabit that were not useful for XGBoost.

### Maintenance: Refactor code for legibility and maintainability
* Unify CPU hist sketching (#5880)
* [R] fix uses of 1:length(x) and other small things (#5992)
* Unify evaluation functions. (#6037)
* Make binary bin search reusable. (#6058)
* Unify set index data. (#6062)
* [R] Remove `stringi` dependency (#6109)
* Merge extract cuts into QuantileContainer. (#6125)
* Reduce C++ compiler warnings (#6197, #6198, #6213, #6286, #6325)
* Cleanup Python code. (#6223)
* Small cleanup to evaluator. (#6400)

### Usability Improvements, Documentation
* [jvm-packages] add example to handle missing value other than 0 (#5677)
* Add DMatrix usage examples to the C API demo (#5854)
* List `DaskDeviceQuantileDMatrix` in the doc. (#5975)
* Update Python custom objective demo. (#5981)
* Update the JSON model schema to document more objective functions. (#5982)
* [Python] Fix warning when `missing` field is not used. (#5969)
* Fix typo in tracker logging (#5994)
* Move a warning about empty dataset, so that it's shown for all objectives and metrics (#5998)
* Fix the instructions for installing the nightly build. (#6004)
* [Doc] Add dtreeviz as a showcase example of integration with 3rd-party software (#6013)
* [jvm-packages] [doc] Update install doc for JVM packages (#6051)
* Fix typo in `xgboost.callback.early_stop` docstring (#6071)
* Add cache suffix to the files used in the external memory demo. (#6088)
* [Doc] Document the parameter `kill_spark_context_on_worker_failure` (#6097)
* Fix link to the demo for custom objectives (#6100)
* Update Dask doc. (#6108)
* Validate weights are positive values. (#6115)
* Document the updated CMake version requirement. (#6123)
* Add demo for `DaskDeviceQuantileDMatrix`. (#6156)
* Cosmetic fixes in `faq.rst` (#6161)
* Fix error message. (#6176)
* [Doc] Add list of winning solutions in data science competitions using XGBoost (#6177)
* Fix a comment in demo to use correct reference (#6190)
* Update the list of winning solutions using XGBoost (#6192)
* Consistent style for build status badge (#6203)
* [Doc] Add info on GPU compiler (#6204)
* Update the list of winning solutions (#6222, #6254)
* Add link to XGBoost's Twitter handle (#6244)
* Fix minor typos in XGBClassifier methods' docstrings (#6247)
* Add sponsors link to FUNDING.yml (#6252)
* Group CLI demo into subdirectory. (#6258)
* Reduce warning messages from `gbtree`. (#6273)
* Create a tutorial for using the C API in a C/C++ application (#6285)
* Update plugin instructions for CMake build (#6289)
* [doc] make Dask distributed example copy-pastable (#6345)
* [Python] Add option to use `libxgboost.so` from the system path (#6362)
* Fixed few grammatical mistakes in doc (#6393)
* Fix broken link in CLI doc (#6396)
* Improve documentation for the Dask API (#6413)
* Revise misleading exception information: no such param of `allow_non_zero_missing` (#6418)
* Fix CLI ranking demo. (#6439)
* Fix broken links. (#6455)

### Acknowledgement
**Contributors**: Nan Zhu (@CodingCat), @FelixYBW, Jack Dunn (@JackDunnNZ), Jean Lescut-Muller (@JeanLescut),  Boris Feld (@Lothiraldan), Nikhil Choudhary (@Nikhil1O1), Rory Mitchell (@RAMitchell), @ShvetsKS, Anthony D'Amato (@Totoketchup), @Wittty-Panda, neko (@akiyamaneko), Alexander Gugel (@alexanderGugel), @dependabot[bot], DIVYA CHAUHAN (@divya661), Daniel Steinberg (@dstein64), Akira Funahashi (@funasoul), Philip Hyunsu Cho (@hcho3), Tong He (@hetong007), Hristo Iliev (@hiliev), Honza Sterba (@honzasterba), @hzy001, Igor Moura (@igormp), @jameskrach, James Lamb (@jameslamb), Naveed Ahmed Saleem Janvekar (@janvekarnaveed), Kyle Nicholson (@kylejn27), lacrosse91 (@lacrosse91), Christian Lorentzen (@lorentzenchr), Manikya Bardhan (@manikyabard), @nabokovas, John Quitto-Graham (@nvidia-johnq), @odidev, Qi Zhang (@qzhang90), Sergio Gavilán (@sgavil), Tanuja Kirthi Doddapaneni (@tanuja3), Cuong Duong (@tcuongd), Yuan Tang (@terrytangyuan), Jiaming Yuan (@trivialfis), vcarpani (@vcarpani), Vladislav Epifanov (@vepifanov), Vitalie Spinu (@vspinu), Bobby Wang (@wbo4958), Zeno Gantner (@zenogantner), zhang_jf (@zuston)

**Reviewers**: Nan Zhu (@CodingCat), John Zedlewski (@JohnZed), Rory Mitchell (@RAMitchell), @ShvetsKS, Egor Smirnov (@SmirnovEgorRu), Anthony D'Amato (@Totoketchup), @Wittty-Panda, Alexander Gugel (@alexanderGugel), Codecov Comments Bot (@codecov-commenter), Codecov (@codecov-io), DIVYA CHAUHAN (@divya661), Devin Robison (@drobison00), Geoffrey Blake (@geoffreyblake), Mark Harris (@harrism), Philip Hyunsu Cho (@hcho3), Honza Sterba (@honzasterba), Igor Moura (@igormp), @jakirkham, @jameskrach, James Lamb (@jameslamb), Janakarajan Natarajan (@janaknat), Jake Hemstad (@jrhemstad), Keith Kraus (@kkraus14), Kyle Nicholson (@kylejn27), Christian Lorentzen (@lorentzenchr), Michael Mayer (@mayer79), Nikolay Petrov (@napetrov), @odidev, PSEUDOTENSOR / Jonathan McKinney (@pseudotensor), Qi Zhang (@qzhang90), Sergio Gavilán (@sgavil), Scott Lundberg (@slundberg), Cuong Duong (@tcuongd), Yuan Tang (@terrytangyuan), Jiaming Yuan (@trivialfis), vcarpani (@vcarpani), Vladislav Epifanov (@vepifanov), Vincent Nijs (@vnijs), Vitalie Spinu (@vspinu), Bobby Wang (@wbo4958), William Hicks (@wphicks)

## v1.2.0 (2020.08.22)

### XGBoost4J-Spark now supports the GPU algorithm (#5171)
* Now XGBoost4J-Spark is able to leverage NVIDIA GPU hardware to speed up training.
* There is on-going work for accelerating the rest of the data pipeline with NVIDIA GPUs (#5950, #5972).

### XGBoost now supports CUDA 11 (#5808)
* It is now possible to build XGBoost with CUDA 11. Note that we do not yet distribute pre-built binaries built with CUDA 11; all current distributions use CUDA 10.0.

### Better guidance for persisting XGBoost models in an R environment (#5940, #5964)
* Users are strongly encouraged to use `xgb.save()` and `xgb.save.raw()` instead of `saveRDS()`. This is so that the persisted models can be accessed with future releases of XGBoost.
* The previous release (1.1.0) had problems loading models that were saved with `saveRDS()`. This release adds a compatibility layer to restore access to the old RDS files. Note that this is meant to be a temporary measure; users are advised to stop using `saveRDS()` and migrate to `xgb.save()` and `xgb.save.raw()`.

### New objectives and metrics
* The pseudo-Huber loss `reg:pseudohubererror` is added (#5647). The corresponding metric is `mphe`. Right now, the slope is hard-coded to 1.
* The Accelerated Failure Time objective for survival analysis (`survival:aft`) is now accelerated on GPUs (#5714, #5716). The survival metrics `aft-nloglik` and `interval-regression-accuracy` are also accelerated on GPUs.

### Improved integration with scikit-learn
* Added `n_features_in_` attribute to the scikit-learn interface to store the number of features used (#5780). This is useful for integrating with some scikit-learn features such as `StackingClassifier`.  See [this link](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html) for more details.
* `XGBoostError` now inherits `ValueError`, which conforms scikit-learn's exception requirement (#5696).

### Improved integration with Dask
* The XGBoost Dask API now exposes an asynchronous interface (#5862). See [the document](https://xgboost.readthedocs.io/en/latest/tutorials/dask.html#working-with-asyncio) for details.
* Zero-copy ingestion of GPU arrays via `DaskDeviceQuantileDMatrix` (#5623, #5799, #5800, #5803, #5837, #5874, #5901): Previously, the Dask interface had to make 2 data copies: one for concatenating the Dask partition/block into a single block and another for internal representation. To save memory, we introduce `DaskDeviceQuantileDMatrix`. As long as Dask partitions are resident in the GPU memory, `DaskDeviceQuantileDMatrix` is able to ingest them directly without making copies. This matrix type wraps `DeviceQuantileDMatrix`.
* The prediction function now returns GPU Series type if the input is from Dask-cuDF (#5710). This is to preserve the input data type.

### Robust handling of external data types (#5689, #5893)
- As we support more and more external data types, the handling logic has proliferated all over the code base and became hard to keep track. It also became unclear how missing values and threads are handled. We refactored the Python package code to collect all data handling logic to a central location, and now we have an explicit list of of all supported data types.

### Improvements in GPU-side data matrix (`DeviceQuantileDMatrix`)
* The GPU-side data matrix now implements its own quantile sketching logic, so that data don't have to be transported back to the main memory (#5700, #5747, #5760, #5846, #5870, #5898). The GK sketching algorithm is also now better documented.
  - Now we can load extremely sparse dataset like URL, although performance is still sub-optimal.
* The GPU-side data matrix now exposes an iterative interface (#5783), so that users are able to construct a matrix from a data iterator. See the [Python demo](https://github.com/dmlc/xgboost/blob/release_1.2.0/demo/guide-python/data_iterator.py).

### New language binding: Swift (#5728)
* Visit https://github.com/kongzii/SwiftXGBoost for more details.

### Robust model serialization with JSON (#5772, #5804, #5831, #5857, #5934)
* We continue efforts from the 1.0.0 release to adopt JSON as the format to save and load models robustly.
* JSON model IO is significantly faster and produces smaller model files.
* Round-trip reproducibility is guaranteed, via the introduction of an efficient float-to-string conversion algorithm known as [the Ryū algorithm](https://dl.acm.org/doi/10.1145/3192366.3192369). The conversion is locale-independent, producing consistent numeric representation regardless of the locale setting of the user's machine.
* We fixed an issue in loading large JSON files to memory.
* It is now possible to load a JSON file from a remote source such as S3.

### Performance improvements
* CPU hist tree method optimization
  - Skip missing lookup in hist row partitioning if data is dense. (#5644)
  - Specialize training procedures for CPU hist tree method on distributed environment. (#5557)
  - Add single point histogram for CPU hist.  Previously gradient histogram for CPU hist is hard coded to be 64 bit, now users can specify the parameter `single_precision_histogram` to use 32 bit histogram instead for faster training performance. (#5624, #5811)
* GPU hist tree method optimization
  - Removed some unnecessary synchronizations and better memory allocation pattern. (#5707)
  - Optimize GPU Hist for wide dataset.  Previously for wide dataset the atomic operation is performed on global memory, now it can run on shared memory for faster histogram building. But there's a known small regression on GeForce cards with dense data. (#5795, #5926, #5948, #5631)

### API additions
* Support passing fmap to importance plot (#5719). Now importance plot can show actual names of features instead of default ones.
* Support 64bit seed. (#5643)
* A new C API `XGBoosterGetNumFeature` is added for getting number of features in booster (#5856).
* Feature names and feature types are now stored in C++ core and saved in binary DMatrix (#5858).

### Breaking: The `predict()` method of `DaskXGBClassifier` now produces class predictions (#5986). Use `predict_proba()` to obtain probability predictions.
* Previously, `DaskXGBClassifier.predict()` produced probability predictions. This is inconsistent with the behavior of other scikit-learn classifiers, where `predict()` returns class predictions. We make a breaking change in 1.2.0 release so that `DaskXGBClassifier.predict()` now correctly produces class predictions and thus behave like other scikit-learn classifiers. Furthermore, we introduce the `predict_proba()` method for obtaining probability predictions, again to be in line with other scikit-learn classifiers.

### Breaking: Custom evaluation metric now receives raw prediction (#5954)
* Previously, the custom evaluation metric received a transformed prediction result when used with a classifier. Now the custom metric will receive a raw (untransformed) prediction and will need to transform the prediction itself.  See [demo/guide-python/custom\_softmax.py](https://github.com/dmlc/xgboost/blob/release_1.2.0/demo/guide-python/custom_softmax.py) for an example.
* This change is to make the custom metric behave consistently with the custom objective, which already receives raw prediction (#5564).

### Breaking: XGBoost4J-Spark now requires Spark 3.0 and Scala 2.12 (#5836, #5890)
* Starting with version 3.0, Spark can manage GPU resources and allocate them among executors.
* Spark 3.0 dropped support for Scala 2.11 and now only supports Scala 2.12. Thus, XGBoost4J-Spark also only supports Scala 2.12.

### Breaking: XGBoost Python package now requires Python 3.6 and later (#5715)
* Python 3.6 has many useful features such as f-strings.

### Breaking: XGBoost now adopts the C++14 standard (#5664)
* Make sure to use a sufficiently modern C++ compiler that supports C++14, such as Visual Studio 2017, GCC 5.0+, and Clang 3.4+.

### Bug-fixes
* Fix a data race in the prediction function (#5853). As a byproduct, the prediction function now uses a thread-local data store and became thread-safe.
* Restore capability to run prediction when the test input has fewer features than the training data (#5955). This capability is necessary to support predicting with LIBSVM inputs. The previous release (1.1) had broken this capability, so we restore it in this version with better tests.
* Fix OpenMP build with CMake for R package, to support CMake 3.13 (#5895).
* Fix Windows 2016 build (#5902, #5918).
* Fix edge cases in scikit-learn interface with Pandas input by disabling feature validation. (#5953)
* [R] Enable weighted learning to rank (#5945)
* [R] Fix early stopping with custom objective (#5923)
* Fix NDK Build (#5886)
* Add missing explicit template specializations for greater portability (#5921)
* Handle empty rows in data iterators correctly (#5929). This bug affects file loader and JVM data frames.
* Fix `IsDense` (#5702)
* [jvm-packages] Fix wrong method name `setAllowZeroForMissingValue` (#5740)
* Fix shape inference for Dask predict (#5989)

### Usability Improvements, Documentation
* [Doc] Document that CUDA 10.0 is required (#5872)
* Refactored command line interface (CLI). Now CLI is able to handle user errors and output basic document. (#5574)
* Better error handling in Python: use `raise from` syntax to preserve full stacktrace (#5787).
* The JSON model dump now has a formal schema (#5660, #5818). The benefit is to prevent `dump_model()` function from breaking. See [this document](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html#difference-between-saving-model-and-dumping-model) to understand the difference between saving and dumping models.
* Add a reference to the GPU external memory paper (#5684)
* Document more objective parameters in the R package (#5682)
* Document the existence of pre-built binary wheels for MacOS (#5711)
* Remove `max.depth` in the R gblinear example. (#5753)
* Added conda environment file for building docs (#5773)
* Mention dask blog post in the doc, which introduces using Dask with GPU and some internal workings. (#5789)
* Fix rendering of Markdown docs (#5821)
* Document new objectives and metrics available on GPUs (#5909)
* Better message when no GPU is found. (#5594)
* Remove the use of `silent` parameter from R demos. (#5675)
* Don't use masked array in array interface. (#5730)
* Update affiliation of @terrytangyuan: Ant Financial -> Ant Group (#5827)
* Move dask tutorial closer other distributed tutorials (#5613)
* Update XGBoost + Dask overview documentation (#5961)
* Show `n_estimators` in the docstring of the scikit-learn interface (#6041)
* Fix a type in a doctring of the scikit-learn interface (#5980)

### Maintenance: testing, continuous integration, build system
* [CI] Remove CUDA 9.0 from CI (#5674, #5745)
* Require CUDA 10.0+ in CMake build (#5718)
* [R] Remove dependency on gendef for Visual Studio builds (fixes #5608) (#5764). This enables building XGBoost with GPU support with R 4.x.
* [R-package] Reduce duplication in configure.ac (#5693)
* Bump com.esotericsoftware to 4.0.2 (#5690)
* Migrate some tests from AppVeyor to GitHub Actions to speed up the tests. (#5911, #5917, #5919, #5922, #5928)
* Reduce cost of the Jenkins CI server (#5884, #5904, #5892). We now enforce a daily budget via an automated monitor. We also dramatically reduced the workload for the Windows platform, since the cloud VM cost is vastly greater for Windows.
* [R] Set up automated R linter (#5944)
* [R] replace uses of T and F with TRUE and FALSE (#5778)
* Update Docker container 'CPU' (#5956)
* Simplify CMake build with modern CMake techniques (#5871)
* Use `hypothesis` package for testing (#5759, #5835, #5849).
* Define `_CRT_SECURE_NO_WARNINGS` to remove unneeded warnings in MSVC (#5434)
* Run all Python demos in CI, to ensure that they don't break (#5651)
* Enhance nvtx support (#5636). Now we can use unified timer between CPU and GPU. Also CMake is able to find nvtx automatically.
* Speed up python test. (#5752)
* Add helper for generating batches of data. (#5756)
* Add c-api-demo to .gitignore (#5855)
* Add option to enable all compiler warnings in GCC/Clang (#5897)
* Make Python model compatibility test runnable locally (#5941)
* Add cupy to Windows CI (#5797)
* [CI] Fix cuDF install; merge 'gpu' and 'cudf' test suite (#5814)
* Update rabit submodule (#5680, #5876)
* Force colored output for Ninja build. (#5959)
* [CI] Assign larger /dev/shm to NCCL (#5966)
* Add missing Pytest marks to AsyncIO unit test (#5968)
* [CI] Use latest cuDF and dask-cudf (#6048)
* Add CMake flag to log C API invocations, to aid debugging (#5925)
* Fix a unit test on CLI, to handle RC versions (#6050)
* [CI] Use mgpu machine to run gpu hist unit tests (#6050)
* [CI] Build GPU-enabled JAR artifact and deploy to xgboost-maven-repo (#6050)

### Maintenance: Refactor code for legibility and maintainability
* Remove dead code in DMatrix initialization. (#5635)
* Catch dmlc error by ref. (#5678)
* Refactor the `gpu_hist` split evaluation in preparation for batched nodes enumeration. (#5610)
* Remove column major specialization. (#5755)
* Remove unused imports in Python (#5776)
* Avoid including `c_api.h` in header files. (#5782)
* Remove unweighted GK quantile, which is unused. (#5816)
* Add Python binding for rabit ops. (#5743)
* Implement `Empty` method for host device vector. (#5781)
* Remove print (#5867)
* Enforce tree order in JSON (#5974)

### Acknowledgement
**Contributors**: Nan Zhu (@CodingCat), @LionOrCatThatIsTheQuestion, Dmitry Mottl (@Mottl), Rory Mitchell (@RAMitchell), @ShvetsKS, Alex Wozniakowski (@a-wozniakowski), Alexander Gugel (@alexanderGugel), @anttisaukko, @boxdot, Andy Adinets (@canonizer), Ram Rachum (@cool-RR), Elliot Hershberg (@elliothershberg), Jason E. Aten, Ph.D. (@glycerine), Philip Hyunsu Cho (@hcho3), @jameskrach, James Lamb (@jameslamb), James Bourbeau (@jrbourbeau), Peter Jung (@kongzii), Lorenz Walthert (@lorenzwalthert), Oleksandr Kuvshynov (@okuvshynov), Rong Ou (@rongou), Shaochen Shi (@shishaochen), Yuan Tang (@terrytangyuan), Jiaming Yuan (@trivialfis), Bobby Wang (@wbo4958), Zhang Zhang (@zhangzhang10)

**Reviewers**: Nan Zhu (@CodingCat), @LionOrCatThatIsTheQuestion, Hao Yang (@QuantHao), Rory Mitchell (@RAMitchell), @ShvetsKS, Egor Smirnov (@SmirnovEgorRu), Alex Wozniakowski (@a-wozniakowski), Amit Kumar (@aktech), Avinash Barnwal (@avinashbarnwal), @boxdot, Andy Adinets (@canonizer), Chandra Shekhar Reddy (@chandrureddy), Ram Rachum (@cool-RR), Cristiano Goncalves (@cristianogoncalves), Elliot Hershberg (@elliothershberg), Jason E. Aten, Ph.D. (@glycerine), Philip Hyunsu Cho (@hcho3), Tong He (@hetong007), James Lamb (@jameslamb), James Bourbeau (@jrbourbeau), Lee Drake (@leedrake5), DougM (@mengdong), Oleksandr Kuvshynov (@okuvshynov), RongOu (@rongou), Shaochen Shi (@shishaochen), Xu Xiao (@sperlingxx), Yuan Tang (@terrytangyuan), Theodore Vasiloudis (@thvasilo), Jiaming Yuan (@trivialfis), Bobby Wang (@wbo4958), Zhang Zhang (@zhangzhang10)

## v1.1.1 (2020.06.06)
This patch release applies the following patches to 1.1.0 release:

* CPU performance improvement in the PyPI wheels (#5720)
* Fix loading old model (#5724)
* Install pkg-config file (#5744)

## v1.1.0 (2020.05.17)

### Better performance on multi-core CPUs (#5244, #5334, #5522)
* Poor performance scaling of the `hist` algorithm for multi-core CPUs has been under investigation (#3810). #5244 concludes the ongoing effort to improve performance scaling on multi-CPUs, in particular Intel CPUs. Roadmap: #5104
* #5334 makes steps toward reducing memory consumption for the `hist` tree method on CPU.
* #5522 optimizes random number generation for data sampling.

### Deterministic GPU algorithm for regression and classification (#5361)
* GPU algorithm for regression and classification tasks is now deterministic.
* Roadmap: #5023. Currently only single-GPU training is deterministic. Distributed training with multiple GPUs is not yet deterministic.

### Improve external memory support on GPUs (#5093, #5365)
* Starting from 1.0.0 release, we added support for external memory on GPUs to enable training with larger datasets. Gradient-based sampling (#5093) speeds up the external memory algorithm by intelligently sampling a subset of the training data to copy into the GPU memory. [Learn more about out-of-core GPU gradient boosting.](https://arxiv.org/abs/2005.09148)
* GPU-side data sketching now works with data from external memory (#5365).

### Parameter validation: detection of unused or incorrect parameters (#5477, #5569, #5508)
* Mis-spelled training parameter is a common user mistake. In previous versions of XGBoost, mis-spelled parameters were silently ignored. Starting with 1.0.0 release, XGBoost will produce a warning message if there is any unused training parameters. The 1.1.0 release makes parameter validation available to the scikit-learn interface (#5477) and the R binding (#5569).

### Thread-safe, in-place prediction method (#5389, #5512)
* Previously, the prediction method was not thread-safe (#5339). This release adds a new API function `inplace_predict()` that is thread-safe. It is now possible to serve concurrent requests for prediction using a shared model object.
* It is now possible to compute prediction in-place for selected data formats (`numpy.ndarray` / `scipy.sparse.csr_matrix` / `cupy.ndarray` / `cudf.DataFrame` / `pd.DataFrame`) without creating a `DMatrix` object.

### Addition of Accelerated Failure Time objective for survival analysis (#4763, #5473, #5486, #5552, #5553)
* Survival analysis (regression) models the time it takes for an event of interest to occur. The target label is potentially censored, i.e. the label is a range rather than a single number. We added a new objective `survival:aft` to support survival analysis. Also added is the new API to specify the ranged labels. Check out [the tutorial](https://xgboost.readthedocs.io/en/release_1.1.0/tutorials/aft_survival_analysis.html) and the [demos](https://github.com/dmlc/xgboost/tree/release_1.1.0/demo/aft_survival).
* GPU support is work in progress (#5714).

### Improved installation experience on Mac OSX (#5597, #5602, #5606, #5701)
* It only takes two commands to install the XGBoost Python package: `brew install libomp` followed by `pip install xgboost`. The installed XGBoost will use all CPU cores. Even better, starting with this release, we distribute pre-compiled binary wheels targeting Mac OSX. Now the install command `pip install xgboost` finishes instantly, as it no longer compiles the C++ source of XGBoost. The last three Mac versions (High Sierra, Mojave, Catalina) are supported.
* R package: the 1.1.0 release fixes the error `Initializing libomp.dylib, but found libomp.dylib already initialized` (#5701)

### Ranking metrics are now accelerated on GPUs (#5380, #5387, #5398)

### GPU-side data matrix to ingest data directly from other GPU libraries (#5420, #5465)
* Previously, data on GPU memory had to be copied back to the main memory before it could be used by XGBoost. Starting with 1.1.0 release, XGBoost provides a dedicated interface (`DeviceQuantileDMatrix`) so that it can ingest data from GPU memory directly. The result is that XGBoost interoperates better with GPU-accelerated data science libraries, such as cuDF, cuPy, and PyTorch.
* Set device in device dmatrix. (#5596)

### Robust model serialization with JSON (#5123, #5217)
* We continue efforts from the 1.0.0 release to adopt JSON as the format to save and load models robustly. Refer to the release note for 1.0.0 to learn more.
* It is now possible to store internal configuration of the trained model (`Booster`) object in R as a JSON string (#5123, #5217).

### Improved integration with Dask
* Pass through `verbose` parameter for dask fit (#5413)
* Use `DMLC_TASK_ID`. (#5415)
* Order the prediction result. (#5416)
* Honor `nthreads` from dask worker. (#5414)
* Enable grid searching with scikit-learn. (#5417)
* Check non-equal when setting threads. (#5421)
* Accept other inputs for prediction. (#5428)
* Fix missing value for scikit-learn interface. (#5435)

### XGBoost4J-Spark: Check number of columns in the data iterator (#5202, #5303)
* Before, the native layer in XGBoost did not know the number of columns (features) ahead of time and had to guess the number of columns by counting the feature index when ingesting data. This method has a failure more in distributed setting: if the training data is highly sparse, some features may be completely missing in one or more worker partitions. Thus, one or more workers may deduce an incorrect data shape, leading to crashes or silently wrong models.
* Enforce correct data shape by passing the number of columns explicitly from the JVM layer into the native layer.

### Major refactoring of the `DMatrix` class
* Continued from 1.0.0 release.
* Remove update prediction cache from predictors. (#5312)
* Predict on Ellpack. (#5327)
* Partial rewrite EllpackPage (#5352)
* Use ellpack for prediction only when sparsepage doesn't exist. (#5504)
* RFC: #4354, Roadmap: #5143

### Breaking: XGBoost Python package now requires Pip 19.0 and higher (#5589)
* Your Linux machine may have an old version of Pip and may attempt to install a source package, leading to long installation time. This is because we are now using `manylinux2010` tag in the binary wheel release. Ensure you have Pip 19.0 or newer by running `python3 -m pip -V` to check the version. Upgrade Pip with command
```
python3 -m pip install --upgrade pip
```
Upgrading to latest pip allows us to depend on newer versions of system libraries. [TensorFlow](https://www.tensorflow.org/install/pip) also requires Pip 19.0+.

### Breaking: GPU algorithm now requires CUDA 10.0 and higher (#5649)
* CUDA 10.0 is necessary to make the GPU algorithm deterministic (#5361).

### Breaking: `silent` parameter is now removed (#5476)
* Please use `verbosity` instead.

### Breaking: Set `output_margin` to True for custom objectives (#5564)
* Now both R and Python interface custom objectives get un-transformed (raw) prediction outputs.

### Breaking: `Makefile` is now removed. We use CMake exclusively to build XGBoost (#5513)
* Exception: the R package uses Autotools, as the CRAN ecosystem did not yet adopt CMake widely.

### Breaking: `distcol` updater is now removed (#5507)
* The `distcol` updater has been long broken, and currently we lack resources to implement a working implementation from scratch.

### Deprecation notices
* **Python 3.5**. This release is the last release to support Python 3.5. The following release (1.2.0) will require Python 3.6.
* **Scala 2.11**. Currently XGBoost4J supports Scala 2.11. However, if a future release of XGBoost adopts Spark 3, it will not support Scala 2.11, as Spark 3 requires Scala 2.12+. We do not yet know which XGBoost release will adopt Spark 3.

### Known limitations
* (Python package) When early stopping is activated with `early_stopping_rounds` at training time, the prediction method (`xgb.predict()`) behaves in a surprising way. If XGBoost runs for M rounds and chooses iteration N (N < M) as the best iteration, then the prediction method will use M trees by default. To use the best iteration (N trees), users will need to manually take the best iteration field `bst.best_iteration` and pass it as the `ntree_limit` argument to `xgb.predict()`. See #5209 and #4052 for additional context.
* GPU ranking objective is currently not deterministic (#5561).
* When training parameter `reg_lambda` is set to zero, some leaf nodes may be assigned a NaN value. (See [discussion](https://discuss.xgboost.ai/t/still-getting-unexplained-nans-new-replication-code/1383/9).) For now, please set `reg_lambda` to a nonzero value.

### Community and Governance
* The XGBoost Project Management Committee (PMC) is pleased to announce a new committer: Egor Smirnov (@SmirnovEgorRu). He has led a major initiative to improve the performance of XGBoost on multi-core CPUs.

### Bug-fixes
* Improved compatibility with scikit-learn (#5255, #5505, #5538)
* Remove f-string, since it's not supported by Python 3.5 (#5330). Note that Python 3.5 support is deprecated and schedule to be dropped in the upcoming release (1.2.0).
* Fix the pruner so that it doesn't prune the same branch twice (#5335)
* Enforce only major version in JSON model schema (#5336). Any major revision of the model schema would bump up the major version.
* Fix a small typo in sklearn.py that broke multiple eval metrics (#5341)
* Restore loading model from a memory buffer (#5360)
* Define lazy isinstance for Python compat (#5364)
* [R] fixed uses of `class()` (#5426)
* Force compressed buffer to be 4 bytes aligned, to keep cuda-memcheck happy (#5441)
* Remove warning for calling host function (`std::max`) on a GPU device (#5453)
* Fix uninitialized value bug in xgboost callback (#5463)
* Fix model dump in CLI (#5485)
* Fix out-of-bound array access in `WQSummary::SetPrune()` (#5493)
* Ensure that configured `dmlc/build_config.h` is picked up by Rabit and XGBoost, to fix build on Alpine (#5514)
* Fix a misspelled method, made in a git merge (#5509)
* Fix a bug in binary model serialization (#5532)
* Fix CLI model IO (#5535)
* Don't use `uint` for threads (#5542)
* Fix R interaction constraints to handle more than 100000 features (#5543)
* [jvm-packages] XGBoost Spark should deal with NaN when parsing evaluation output (#5546)
* GPU-side data sketching is now aware of query groups in learning-to-rank data (#5551)
* Fix DMatrix slicing for newly added fields (#5552)
* Fix configuration status with loading binary model (#5562)
* Fix build when OpenMP is disabled (#5566)
* R compatibility patches (#5577, #5600)
* gpu\_hist performance fixes (#5558)
* Don't set seed on CLI interface (#5563)
* [R] When serializing model, preserve model attributes related to early stopping (#5573)
* Avoid rabit calls in learner configuration (#5581)
* Hide C++ symbols in libxgboost.so when building Python wheel (#5590). This fixes apache/incubator-tvm#4953.
* Fix compilation on Mac OSX High Sierra (10.13) (#5597)
* Fix build on big endian CPUs (#5617)
* Resolve crash due to use of `vector<bool>::iterator` (#5642)
* Validation JSON model dump using JSON schema (#5660)

### Performance improvements
* Wide dataset quantile performance improvement (#5306)
* Reduce memory usage of GPU-side data sketching (#5407)
* Reduce span check overhead (#5464)
* Serialise booster after training to free up GPU memory (#5484)
* Use the maximum amount of GPU shared memory available to speed up the histogram kernel (#5491)
* Use non-synchronising scan in Thrust (#5560)
* Use `cudaDeviceGetAttribute()` instead of `cudaGetDeviceProperties()` for speed (#5570)

### API changes
* Support importing data from a Pandas SparseArray (#5431)
* `HostDeviceVector` (vector shared between CPU and GPU memory) now exposes `HostSpan` interface, to enable access on the CPU side with bound check (#5459)
* Accept other gradient types for `SplitEntry` (#5467)

### Usability Improvements, Documentation
* Add `JVM_CHECK_CALL` to prevent C++ exceptions from leaking into the JVM layer (#5199)
* Updated Windows build docs (#5283)
* Update affiliation of @hcho3 (#5292)
* Display Sponsor button, link to OpenCollective (#5325)
* Update docs for GPU external memory (#5332)
* Add link to GPU documentation (#5437)
* Small updates to GPU documentation (#5483)
* Edits on tutorial for XGBoost job on Kubernetes (#5487)
* Add reference to GPU external memory (#5490)
* Fix typos (#5346, #5371, #5384, #5399, #5482, #5515)
* Update Python doc (#5517)
* Add Neptune and Optuna to list of examples (#5528)
* Raise error if the number of data weights doesn't match the number of data sets (#5540)
* Add a note about GPU ranking (#5572)
* Clarify meaning of `training` parameter in the C API function `XGBoosterPredict()` (#5604)
* Better error handling for situations where existing trees cannot be modified (#5406, #5418). This feature is enabled when `process_type` is set to `update`.

### Maintenance: testing, continuous integration, build system
* Add C++ test coverage for data sketching (#5251)
* Ignore gdb\_history (#5257)
* Rewrite setup.py. (#5271, #5280)
* Use `scikit-learn` in extra dependencies (#5310)
* Add CMake option to build static library (#5397)
* [R] changed FindLibR to take advantage of CMake cache (#5427)
* [R] fixed inconsistency in R -e calls in FindLibR.cmake (#5438)
* Refactor tests with data generator (#5439)
* Resolve failing Travis CI (#5445)
* Update dmlc-core. (#5466)
* [CI] Use clang-tidy 10 (#5469)
* De-duplicate code for checking maximum number of nodes (#5497)
* [CI] Use Ubuntu 18.04 LTS in JVM CI, because 19.04 is EOL (#5537)
* [jvm-packages] [CI] Create a Maven repository to host SNAPSHOT JARs (#5533)
* [jvm-packages] [CI] Publish XGBoost4J JARs with Scala 2.11 and 2.12 (#5539)
* [CI] Use Vault repository to re-gain access to devtoolset-4 (#5589)

### Maintenance: Refactor code for legibility and maintainability
* Move prediction cache to Learner (#5220, #5302)
* Remove SimpleCSRSource (#5315)
* Refactor SparsePageSource, delete cache files after use (#5321)
* Remove unnecessary DMatrix methods (#5324)
* Split up `LearnerImpl` (#5350)
* Move segment sorter to common (#5378)
* Move thread local entry into Learner (#5396)
* Split up test helpers header (#5455)
* Requires setting leaf stat when expanding tree (#5501)
* Purge device\_helpers.cuh (#5534)
* Use thrust functions instead of custom functions (#5544)

### Acknowledgement
**Contributors**: Nan Zhu (@CodingCat), Rory Mitchell (@RAMitchell), @ShvetsKS, Egor Smirnov (@SmirnovEgorRu), Andrew Kane (@ankane), Avinash Barnwal (@avinashbarnwal), Bart Broere (@bartbroere), Andy Adinets (@canonizer), Chen Qin (@chenqin), Daiki Katsuragawa (@daikikatsuragawa), David Díaz Vico (@daviddiazvico), Darius Kharazi (@dkharazi), Darby Payne (@dpayne), Jason E. Aten, Ph.D. (@glycerine), Philip Hyunsu Cho (@hcho3), James Lamb (@jameslamb), Jan Borchmann (@jborchma), Kamil A. Kaczmarek (@kamil-kaczmarek), Melissa Kohl (@mjkohl32), Nicolas Scozzaro (@nscozzaro), Paul Kaefer (@paulkaefer), Rong Ou (@rongou), Samrat Pandiri (@samratp), Sriram Chandramouli (@sriramch), Yuan Tang (@terrytangyuan), Jiaming Yuan (@trivialfis), Liang-Chi Hsieh (@viirya), Bobby Wang (@wbo4958), Zhang Zhang (@zhangzhang10),

**Reviewers**: Nan Zhu (@CodingCat), @LeZhengThu, Rory Mitchell (@RAMitchell), @ShvetsKS, Egor Smirnov (@SmirnovEgorRu), Steve Bronder (@SteveBronder), Nikita Titov (@StrikerRUS), Andrew Kane (@ankane), Avinash Barnwal (@avinashbarnwal), @brydag, Andy Adinets (@canonizer), Chandra Shekhar Reddy (@chandrureddy), Chen Qin (@chenqin), Codecov (@codecov-io), David Díaz Vico (@daviddiazvico), Darby Payne (@dpayne), Jason E. Aten, Ph.D. (@glycerine), Philip Hyunsu Cho (@hcho3), James Lamb (@jameslamb), @johnny-cat, Mu Li (@mli), Mate Soos (@msoos), @rnyak, Rong Ou (@rongou), Sriram Chandramouli (@sriramch), Toby Dylan Hocking (@tdhock), Yuan Tang (@terrytangyuan), Oleksandr Pryimak (@trams), Jiaming Yuan (@trivialfis), Liang-Chi Hsieh (@viirya), Bobby Wang (@wbo4958),

## v1.0.2 (2020.03.03)
This patch release applies the following patches to 1.0.0 release:

* Fix a small typo in sklearn.py that broke multiple eval metrics (#5341)
* Restore loading model from buffer (#5360)
* Use type name for data type check (#5364)

## v1.0.1 (2020.02.21)
This release is identical to the 1.0.0 release, except that it fixes a small bug that rendered 1.0.0 incompatible with Python 3.5. See #5328.

## v1.0.0 (2020.02.19)
This release marks a major milestone for the XGBoost project.

### Apache-style governance, contribution policy, and semantic versioning (#4646, #4659)
* Starting with 1.0.0 release, the XGBoost Project is adopting Apache-style governance. The full community guideline is [available in the doc website](https://xgboost.readthedocs.io/en/release_1.0.0/contrib/community.html). Note that we now have Project Management Committee (PMC) who would steward the project on the long-term basis. The PMC is also entrusted to run and fund the project's continuous integration (CI) infrastructure (https://xgboost-ci.net).
* We also adopt the [semantic versioning](https://semver.org/). See [our release versioning policy](https://xgboost.readthedocs.io/en/release_1.0.0/contrib/release.html).

### Better performance scaling for multi-core CPUs (#4502, #4529, #4716, #4851, #5008, #5107, #5138, #5156)
* Poor performance scaling of the `hist` algorithm for multi-core CPUs has been under investigation (#3810). Previous effort #4529 was replaced with a series of pull requests (#5107, #5138, #5156) aimed at achieving the same performance benefits while keeping the C++ codebase legible. The latest performance benchmark results show [up to 5x speedup on Intel CPUs with many cores](https://github.com/dmlc/xgboost/pull/5156#issuecomment-580024413). Note: #5244, which concludes the effort, will become part of the upcoming release 1.1.0.

### Improved installation experience on Mac OSX (#4672, #5074, #5080, #5146, #5240)
* It used to be quite complicated to install XGBoost on Mac OSX. XGBoost uses OpenMP to distribute work among multiple CPU cores, and Mac's default C++ compiler (Apple Clang) does not come with OpenMP. Existing work-around (using another C++ compiler) was complex and prone to fail with cryptic diagnosis (#4933, #4949, #4969).
* Now it only takes two commands to install XGBoost: `brew install libomp` followed by `pip install xgboost`. The installed XGBoost will use all CPU cores.
* Even better, XGBoost is now available from Homebrew: `brew install xgboost`. See Homebrew/homebrew-core#50467.
* Previously, if you installed the XGBoost R package using the command `install.packages('xgboost')`, it could only use a single CPU core and you would experience slow training performance. With 1.0.0 release, the R package will use all CPU cores out of box.

### Distributed XGBoost now available on Kubernetes (#4621, #4939)
* Check out the [tutorial for setting up distributed XGBoost on a Kubernetes cluster](https://xgboost.readthedocs.io/en/release_1.0.0/tutorials/kubernetes.html).

### Ruby binding for XGBoost (#4856)

### New Native Dask interface for multi-GPU and multi-node scaling (#4473, #4507, #4617, #4819, #4907, #4914, #4941, #4942, #4951, #4973, #5048, #5077, #5144, #5270)
* XGBoost now integrates seamlessly with [Dask](https://dask.org/), a lightweight distributed framework for data processing. Together with the first-class support for cuDF data frames (see below), it is now easier than ever to create end-to-end data pipeline running on one or more NVIDIA GPUs.
* Multi-GPU training with Dask is now up to 20% faster than the previous release (#4914, #4951).

### First-class support for cuDF data frames and cuPy arrays (#4737, #4745, #4794, #4850, #4891, #4902, #4918, #4927, #4928, #5053, #5189, #5194, #5206, #5219, #5225)
* [cuDF](https://github.com/rapidsai/cudf) is a data frame library for loading and processing tabular data on NVIDIA GPUs. It provides a Pandas-like API.
* [cuPy](https://github.com/cupy/cupy) implements a NumPy-compatible multi-dimensional array on NVIDIA GPUs.
* Now users can keep the data on the GPU memory throughout the end-to-end data pipeline, obviating the need for copying data between the main memory and GPU memory.
* XGBoost can accept any data structure that exposes `__array_interface__` signature, opening way to support other columar formats that are compatible with Apache Arrow.

### [Feature interaction constraint](https://xgboost.readthedocs.io/en/release_1.0.0/tutorials/feature_interaction_constraint.html) is now available with `approx` and `gpu_hist` algorithms (#4534, #4587, #4596, #5034).

### Learning to rank is now GPU accelerated (#4873, #5004, #5129)
* Supported ranking objectives: NDGC, Map, Pairwise.
* [Up to 2x improved training performance on GPUs](https://devblogs.nvidia.com/learning-to-rank-with-xgboost-and-gpu/).

### Enable `gamma` parameter for GPU training (#4874, #4953)
* The `gamma` parameter specifies the minimum loss reduction required to add a new split in a tree. A larger value for `gamma` has the effect of pre-pruning the tree, by making harder to add splits.

### External memory for GPU training (#4486, #4526, #4747, #4833, #4879, #5014)
* It is now possible to use NVIDIA GPUs even when the size of training data exceeds the available GPU memory. Note that the external memory support for GPU is still experimental. #5093 will further improve performance and will become part of the upcoming release 1.1.0.
* RFC for enabling external memory with GPU algorithms: #4357

### Improve Scikit-Learn interface (#4558, #4842, #4929, #5049, #5151, #5130, #5227)
* Many users of XGBoost enjoy the convenience and breadth of Scikit-Learn ecosystem. In this release, we revise the Scikit-Learn API of XGBoost (`XGBRegressor`, `XGBClassifier`, and `XGBRanker`) to achieve feature parity with the traditional XGBoost interface (`xgboost.train()`).
* Insert check to validate data shapes.
* Produce an error message if `eval_set` is not a tuple. An error message is better than silently crashing.
* Allow using `numpy.RandomState` object.
* Add `n_jobs` as an alias of `nthread`.
* Roadmap: #5152

### XGBoost4J-Spark: Redesigning checkpointing mechanism
* RFC is available at #4786
* Clean up checkpoint file after a successful training job (#4754): The current implementation in XGBoost4J-Spark does not clean up the checkpoint file after a successful training job. If the user runs another job with the same checkpointing directory, she will get a wrong model because the second job will re-use the checkpoint file left over from the first job. To prevent this scenario, we propose to always clean up the checkpoint file after every successful training job.
* Avoid Multiple Jobs for Checkpointing (#5082): The current method for checkpoint is to collect the booster produced at the last iteration of each checkpoint internal to Driver and persist it in HDFS. The major issue with this approach is that it needs to re-perform the data preparation for training if the user did not choose to cache the training dataset. To avoid re-performing data prep, we build external-memory checkpointing in the XGBoost4J layer as well.
* Enable deterministic repartitioning when checkpoint is enabled (#4807): Distributed algorithm for gradient boosting assumes a fixed partition of the training data between multiple iterations. In previous versions, there was no guarantee that data partition would stay the same, especially when a worker goes down and some data had to recovered from previous checkpoint. In this release, we make data partition deterministic by using the data hash value of each data row in computing the partition.

### XGBoost4J-Spark: handle errors thrown by the native code (#4560)
* All core logic of XGBoost is written in C++, so XGBoost4J-Spark internally uses the C++ code via Java Native Interface (JNI). #4560 adds a proper error handling for any errors or exceptions arising from the C++ code, so that the XGBoost Spark application can be torn down in an orderly fashion.

### XGBoost4J-Spark: Refine method to count the number of alive cores  (#4858)
* The `SparkParallelismTracker` class ensures that sufficient number of executor cores are alive. To that end, it is important to query the number of alive cores reliably.

### XGBoost4J: Add `BigDenseMatrix` to store more than `Integer.MAX_VALUE` elements (#4383)

### Robust model serialization with JSON (#4632, #4708, #4739, #4868, #4936, #4945, #4974, #5086, #5087, #5089, #5091, #5094, #5110, #5111, #5112, #5120, #5137, #5218, #5222, #5236, #5245, #5248, #5281)
* In this release, we introduce an experimental support of using [JSON](https://www.json.org/json-en.html) for serializing (saving/loading) XGBoost models and related hyperparameters for training. We would like to eventually replace the old binary format with JSON, since it is an open format and parsers are available in many programming languages and platforms. See [the documentation for model I/O using JSON](https://xgboost.readthedocs.io/en/release_1.0.0/tutorials/saving_model.html). #3980 explains why JSON was chosen over other alternatives.
* To maximize interoperability and compatibility of the serialized models, we now split serialization into two parts (#4855):
  1. Model, e.g. decision trees and strictly related metadata like `num_features`.
  2. Internal configuration, consisting of training parameters and other configurable parameters. For example, `max_delta_step`, `tree_method`, `objective`, `predictor`, `gpu_id`.

  Previously, users often ran into issues where the model file produced by one machine could not load or run on another machine. For example, models trained using a machine with an NVIDIA GPU could not run on another machine without a GPU (#5291, #5234). The reason is that the old binary format saved some internal configuration that were not universally applicable to all machines, e.g. `predictor='gpu_predictor'`.

  Now, model saving function (`Booster.save_model()` in Python) will save only the model, without internal configuration. This will guarantee that your model file would be used anywhere. Internal configuration will be serialized in limited circumstances such as:
  * Multiple nodes in a distributed system exchange model details over the network.
  * Model checkpointing, to recover from possible crashes.

  This work proved to be useful for parameter validation as well (see below).
* Starting with 1.0.0 release, we will use semantic versioning to indicate whether the model produced by one version of XGBoost would be compatible with another version of XGBoost. Any change in the major version indicates a breaking change in the serialization format.
* We now provide a robust method to save and load scikit-learn related attributes (#5245). Previously, we used Python pickle to save Python attributes related to `XGBClassifier`, `XGBRegressor`, and `XGBRanker` objects. The attributes are necessary to properly interact with scikit-learn. See #4639 for more details. The use of pickling hampered interoperability, as a pickle from one machine may not necessarily work on another machine. Starting with this release, we use an alternative method to serialize the scikit-learn related attributes. The use of Python pickle is now discouraged (#5236, #5281).

### Parameter validation: detection of unused or incorrect parameters (#4553, #4577, #4738, #4801, #4961, #5101, #5157, #5167, #5256)
* Mis-spelled training parameter is a common user mistake. In previous versions of XGBoost, mis-spelled parameters were silently ignored. Starting with 1.0.0 release, XGBoost will produce a warning message if there is any unused training parameters. Currently, parameter validation is available to R users and Python XGBoost API users. We are working to extend its support to scikit-learn users.
* Configuration steps now have well-defined semantics (#4542, #4738), so we know exactly where and how the internal configurable parameters are changed.
* The user can now use `save_config()` function to inspect all (used) training parameters. This is helpful for debugging model performance.

### Allow individual workers to recover from faults (#4808, #4966)
* Status quo: if a worker fails, all workers are shut down and restarted, and learning resumes from the last checkpoint. This involves requesting resources from the scheduler (e.g. Spark) and shuffling all the data again from scratch. Both of these operations can be quite costly and block training for extended periods of time, especially if the training data is big and the number of worker nodes is in the hundreds.
* The proposed solution is to recover the single node that failed, instead of shutting down all workers. The rest of the clusters wait until the single failed worker is bootstrapped and catches up with the rest.
* See roadmap at #4753. Note that this is work in progress. In particular, the feature is not yet available from XGBoost4J-Spark.

### Accurate prediction for DART models
* Use DART tree weights when computing SHAPs (#5050)
* Don't drop trees during DART prediction by default (#5115)
* Fix DART prediction in R (#5204)

### Make external memory more robust
* Fix issues with training with external memory on cpu (#4487)
* Fix crash with approx tree method on cpu (#4510)
* Fix external memory race in `exact` (#4980). Note: `dmlc::ThreadedIter` is not actually thread-safe. We would like to re-design it in the long term.

### Major refactoring of the `DMatrix` class (#4686, #4744, #4748, #5044, #5092, #5108, #5188, #5198)
* Goal 1: improve performance and reduce memory consumption. Right now, if the user trains a model with a NumPy array as training data, the array gets copies 2-3 times before training begins. We'd like to reduce duplication of the data matrix.
* Goal 2: Expose a common interface to external data, unify the way DMatrix objects are constructed and simplify the process of adding new external data sources. This work is essential for ingesting cuPy arrays.
* Goal 3: Handle missing values consistently.
* RFC: #4354, Roadmap: #5143
* This work is also relevant to external memory support on GPUs.

### Breaking: XGBoost Python package now requires Python 3.5 or newer (#5021, #5274)
* Python 3.4 has reached its end-of-life on March 16, 2019, so we now require Python 3.5 or newer.

### Breaking: GPU algorithm now requires CUDA 9.0 and higher (#4527, #4580)

### Breaking: `n_gpus` parameter removed; multi-GPU training now requires a distributed framework (#4579, #4749, #4773, #4810, #4867, #4908)
* #4531 proposed removing support for single-process multi-GPU training. Contributors would focus on multi-GPU support through distributed frameworks such as Dask and Spark, where the framework would be expected to assign a worker process for each GPU independently. By delegating GPU management and data movement to the distributed framework, we can greatly simplify the core XGBoost codebase, make multi-GPU training more robust, and reduce burden for future development.

### Breaking: Some deprecated features have been removed
* ``gpu_exact`` training method (#4527, #4742, #4777). Use ``gpu_hist`` instead.
* ``learning_rates`` parameter in Python (#5155). Use the callback API instead.
* ``num_roots`` (#5059, #5165), since the current training code always uses a single root node.
* GPU-specific objectives (#4690), such as `gpu:reg:linear`. Use objectives without `gpu:` prefix; GPU will be used automatically if your machine has one.

### Breaking: the C API function `XGBoosterPredict()` now asks for an extra parameter `training`.

### Breaking: We now use CMake exclusively to build XGBoost. `Makefile` is being sunset.
* Exception: the R package uses Autotools, as the CRAN ecosystem did not yet adopt CMake widely.

### Performance improvements
* Smarter choice of histogram construction for distributed `gpu_hist` (#4519)
* Optimizations for quantization on device (#4572)
* Introduce caching memory allocator to avoid latency associated with GPU memory allocation (#4554, #4615)
* Optimize the initialization stage of the CPU `hist` algorithm for sparse datasets (#4625)
* Prevent unnecessary data copies from GPU memory to the host (#4795)
* Improve operation efficiency for single prediction (#5016)
* Group builder modified for incremental building, to speed up building large `DMatrix` (#5098)

### Bug-fixes
* Eliminate `FutureWarning: Series.base is deprecated` (#4337)
* Ensure pandas DataFrame column names are treated as strings in type error message (#4481)
* [jvm-packages] Add back `reg:linear` for scala, as it is only deprecated and not meant to be removed yet (#4490)
* Fix library loading for Cygwin users (#4499)
* Fix prediction from loaded pickle (#4516)
* Enforce exclusion between `pred_interactions=True` and `pred_interactions=True` (#4522)
* Do not return dangling reference to local `std::string` (#4543)
* Set the appropriate device before freeing device memory (#4566)
* Mark `SparsePageDmatrix` destructor default. (#4568)
* Choose the appropriate tree method only when the tree method is 'auto' (#4571)
* Fix `benchmark_tree.py` (#4593)
* [jvm-packages] Fix silly bug in feature scoring (#4604)
* Fix GPU predictor when the test data matrix has different number of features than the training data matrix used to train the model (#4613)
* Fix external memory for get column batches. (#4622)
* [R] Use built-in label when xgb.DMatrix is given to xgb.cv() (#4631)
* Fix early stopping in the Python package (#4638)
* Fix AUC error in distributed mode caused by imbalanced dataset (#4645, #4798)
* [jvm-packages] Expose `setMissing` method in `XGBoostClassificationModel` / `XGBoostRegressionModel` (#4643)
* Remove initializing stringstream reference. (#4788)
* [R] `xgb.get.handle` now checks all class listed of `object` (#4800)
* Do not use `gpu_predictor` unless data comes from GPU (#4836)
* Fix data loading (#4862)
* Workaround `isnan` across different environments. (#4883)
* [jvm-packages] Handle Long-type parameter (#4885)
* Don't `set_params` at the end of `set_state` (#4947). Ensure that the model does not change after pickling and unpickling multiple times.
* C++ exceptions should not crash OpenMP loops (#4960)
* Fix `usegpu` flag in DART. (#4984)
* Run training with empty `DMatrix` (#4990, #5159)
* Ensure that no two processes can use the same GPU (#4990)
* Fix repeated split and 0 cover nodes (#5010)
* Reset histogram hit counter between multiple data batches (#5035)
* Fix `feature_name` crated from int64index dataframe. (#5081)
* Don't use 0 for "fresh leaf" (#5084)
* Throw error when user attempts to use multi-GPU training and XGBoost has not been compiled with NCCL (#5170)
* Fix metric name loading (#5122)
* Quick fix for memory leak in CPU `hist` algorithm (#5153)
* Fix wrapping GPU ID and prevent data copying (#5160)
* Fix signature of Span constructor (#5166)
* Lazy initialization of device vector, so that XGBoost compiled with CUDA can run on a machine without any GPU (#5173)
* Model loading should not change system locale (#5314)
* Distributed training jobs would sometimes hang; revert Rabit to fix this regression (dmlc/rabit#132, #5237)

### API changes
* Add support for cross-validation using query ID (#4474)
* Enable feature importance property for DART model (#4525)
* Add `rmsle` metric and `reg:squaredlogerror` objective (#4541)
* All objective and evaluation metrics are now exposed to JVM packages (#4560)
* `dump_model()` and `get_dump()` now support exporting in GraphViz language (#4602)
* Support metrics `ndcg-` and `map-` (#4635)
* [jvm-packages] Allow chaining prediction (transform) in XGBoost4J-Spark (#4667)
* [jvm-packages] Add option to bypass missing value check in the Spark layer (#4805). Only use this option if you know what you are doing.
* [jvm-packages] Add public group getter (#4838)
* `XGDMatrixSetGroup` C API is now deprecated (#4864). Use `XGDMatrixSetUIntInfo` instead.
* [R] Added new `train_folds` parameter to `xgb.cv()` (#5114)
* Ingest meta information from Pandas DataFrame, such as data weights (#5216)

### Maintenance: Refactor code for legibility and maintainability
* De-duplicate GPU parameters (#4454)
* Simplify INI-style config reader using C++11 STL (#4478, #4521)
* Refactor histogram building code for `gpu_hist` (#4528)
* Overload device memory allocator, to enable instrumentation for compiling memory usage statistics (#4532)
* Refactor out row partitioning logic from `gpu_hist` (#4554)
* Remove an unused variable (#4588)
* Implement tree model dump with code generator, to de-duplicate code for generating dumps in 3 different formats (#4602)
* Remove `RowSet` class which is no longer being used (#4697)
* Remove some unused functions as reported by cppcheck (#4743)
* Mimic CUDA assert output in Span check (#4762)
* [jvm-packages] Refactor `XGBoost.scala` to put all params processing in one place (#4815)
* Add some comments for GPU row partitioner (#4832)
* Span: use `size_t' for index_type,  add `front' and `back'. (#4935)
* Remove dead code in `exact` algorithm (#5034, #5105)
* Unify integer types used for row and column indices (#5034)
* Extract feature interaction constraint from `SplitEvaluator` class. (#5034)
* [Breaking] De-duplicate paramters and docstrings in the constructors of Scikit-Learn models (#5130)
* Remove benchmark code from GPU tests (#5141)
* Clean up Python 2 compatibility code. (#5161)
* Extensible binary serialization format for `DMatrix::MetaInfo` (#5187). This will be useful for implementing censored labels for survival analysis applications.
* Cleanup clang-tidy warnings. (#5247)

### Maintenance: testing, continuous integration, build system
* Use `yaml.safe_load` instead of `yaml.load`. (#4537)
* Ensure GCC is at least 5.x (#4538)
* Remove all mention of `reg:linear` from tests (#4544)
* [jvm-packages] Upgrade to Scala 2.12 (#4574)
* [jvm-packages] Update kryo dependency to 2.22 (#4575)
* [CI] Specify account ID when logging into ECR Docker registry (#4584)
* Use Sphinx 2.1+ to compile documentation (#4609)
* Make Pandas optional for running Python unit tests (#4620)
* Fix spark tests on machines with many cores (#4634)
* [jvm-packages] Update local dev build process (#4640)
* Add optional dependencies to setup.py (#4655)
* [jvm-packages] Fix maven warnings (#4664)
* Remove extraneous files from the R package, to comply with CRAN policy (#4699)
* Remove VC-2013 support, since it is not C++11 compliant (#4701)
* [CI] Fix broken installation of Pandas (#4704, #4722)
* [jvm-packages] Clean up temporary files afer running tests (#4706)
* Specify version macro in CMake. (#4730)
* Include dmlc-tracker into XGBoost Python package (#4731)
* [CI] Use long key ID for Ubuntu repository fingerprints. (#4783)
* Remove plugin, CUDA related code in automake & autoconf files (#4789)
* Skip related tests when scikit-learn is not installed. (#4791)
* Ignore vscode and clion files (#4866)
* Use bundled Google Test by default (#4900)
* [CI] Raise timeout threshold in Jenkins (#4938)
* Copy CMake parameter from dmlc-core. (#4948)
* Set correct file permission. (#4964)
* [CI] Update lint configuration to support latest pylint convention (#4971)
* [CI] Upload nightly builds to S3 (#4976, #4979)
* Add asan.so.5 to cmake script. (#4999)
* [CI] Fix Travis tests. (#5062)
* [CI] Locate vcomp140.dll from System32 directory (#5078)
* Implement training observer to dump internal states of objects (#5088). This will be useful for debugging.
* Fix visual studio output library directories (#5119)
* [jvm-packages] Comply with scala style convention + fix broken unit test (#5134)
* [CI] Repair download URL for Maven 3.6.1 (#5139)
* Don't use modernize-use-trailing-return-type in clang-tidy. (#5169)
* Explicitly use UTF-8 codepage when using MSVC (#5197)
* Add CMake option to run Undefined Behavior Sanitizer (UBSan) (#5211)
* Make some GPU tests deterministic (#5229)
* [R] Robust endian detection in CRAN xgboost build (#5232)
* Support FreeBSD (#5233)
* Make `pip install xgboost*.tar.gz` work by fixing build-python.sh (#5241)
* Fix compilation error due to 64-bit integer narrowing to `size_t` (#5250)
* Remove use of `std::cout` from R package, to comply with CRAN policy (#5261)
* Update DMLC-Core submodule (#4674, #4688, #4726, #4924)
* Update Rabit submodule (#4560, #4667, #4718, #4808, #4966, #5237)

### Usability Improvements, Documentation
* Add Random Forest API to Python API doc (#4500)
* Fix Python demo and doc. (#4545)
* Remove doc about not supporting CUDA 10.1 (#4578)
* Address some sphinx warnings and errors, add doc for building doc. (#4589)
* Add instruction to run formatting checks locally (#4591)
* Fix docstring for `XGBModel.predict()` (#4592)
* Doc and demo for customized metric and objective (#4598, #4608)
* Add to documentation how to run tests locally (#4610)
* Empty evaluation list in early stopping should produce meaningful error message (#4633)
* Fixed year to 2019 in conf.py, helpers.h and LICENSE (#4661)
* Minor updates to links and grammar (#4673)
* Remove `silent` in doc (#4689)
* Remove old Python trouble shooting doc (#4729)
* Add `os.PathLike` support for file paths to DMatrix and Booster Python classes (#4757)
* Update XGBoost4J-Spark doc (#4804)
* Regular formatting for evaluation metrics (#4803)
* [jvm-packages] Refine documentation for handling missing values in XGBoost4J-Spark (#4805)
* Monitor for distributed environment (#4829). This is useful for identifying performance bottleneck.
* Add check for length of weights and produce a good error message (#4872)
* Fix DMatrix doc (#4884)
* Export C++ headers in CMake installation (#4897)
* Update license year in README.md to 2019 (#4940)
* Fix incorrectly displayed Note in the doc (#4943)
* Follow PEP 257 Docstring Conventions (#4959)
* Document minimum version required for Google Test (#5001)
* Add better error message for invalid feature names (#5024)
* Some guidelines on device memory usage (#5038)
* [doc] Some notes for external memory. (#5065)
* Update document for `tree_method` (#5106)
* Update demo for ranking. (#5154)
* Add new lines for Spark XGBoost missing values section (#5180)
* Fix simple typo: utilty -> utility (#5182)
* Update R doc by roxygen2 (#5201)
* [R] Direct user to use `set.seed()` instead of setting `seed` parameter (#5125)
* Add Optuna badge to `README.md` (#5208)
* Fix compilation error in `c-api-demo.c` (#5215)

### Acknowledgement
**Contributors**: Nan Zhu (@CodingCat), Crissman Loomis (@Crissman), Cyprien Ricque (@Cyprien-Ricque), Evan Kepner (@EvanKepner), K.O. (@Hi-king), KaiJin Ji (@KerryJi), Peter Badida (@KeyWeeUsr), Kodi Arfer (@Kodiologist), Rory Mitchell (@RAMitchell), Egor Smirnov (@SmirnovEgorRu), Jacob Kim (@TheJacobKim), Vibhu Jawa (@VibhuJawa), Marcos (@astrowonk), Andy Adinets (@canonizer), Chen Qin (@chenqin), Christopher Cowden (@cowden), @cpfarrell, @david-cortes, Liangcai Li (@firestarman), @fuhaoda, Philip Hyunsu Cho (@hcho3), @here-nagini, Tong He (@hetong007), Michal Kurka (@michalkurka), Honza Sterba (@honzasterba), @iblumin, @koertkuipers, mattn (@mattn), Mingjie Tang (@merlintang), OrdoAbChao (@mglowacki100), Matthew Jones (@mt-jones), mitama (@nigimitama), Nathan Moore (@nmoorenz), Daniel Stahl (@phillyfan1138), Michaël Benesty (@pommedeterresautee), Rong Ou (@rongou), Sebastian (@sfahnens), Xu Xiao (@sperlingxx), @sriramch, Sean Owen (@srowen), Stephanie Yang (@stpyang), Yuan Tang (@terrytangyuan), Mathew Wicks (@thesuperzapper), Tim Gates (@timgates42), TinkleG (@tinkle1129), Oleksandr Pryimak (@trams), Jiaming Yuan (@trivialfis), Matvey Turkov (@turk0v), Bobby Wang (@wbo4958), yage (@yage99), @yellowdolphin

**Reviewers**: Nan Zhu (@CodingCat), Crissman Loomis (@Crissman), Cyprien Ricque (@Cyprien-Ricque), Evan Kepner (@EvanKepner), John Zedlewski (@JohnZed), KOLANICH (@KOLANICH), KaiJin Ji (@KerryJi), Kodi Arfer (@Kodiologist), Rory Mitchell (@RAMitchell), Egor Smirnov (@SmirnovEgorRu), Nikita Titov (@StrikerRUS), Jacob Kim (@TheJacobKim), Vibhu Jawa (@VibhuJawa), Andrew Kane (@ankane), Arno Candel (@arnocandel), Marcos (@astrowonk), Bryan Woods (@bryan-woods), Andy Adinets (@canonizer), Chen Qin (@chenqin), Thomas Franke (@coding-komek), Peter  (@codingforfun), @cpfarrell, Joshua Patterson (@datametrician), @fuhaoda, Philip Hyunsu Cho (@hcho3), Tong He (@hetong007), Honza Sterba (@honzasterba), @iblumin, @jakirkham, Vadim Khotilovich (@khotilov), Keith Kraus (@kkraus14), @koertkuipers, @melonki, Mingjie Tang (@merlintang), OrdoAbChao (@mglowacki100), Daniel Mahler (@mhlr), Matthew Rocklin (@mrocklin), Matthew Jones (@mt-jones), Michaël Benesty (@pommedeterresautee), PSEUDOTENSOR / Jonathan McKinney (@pseudotensor), Rong Ou (@rongou), Vladimir (@sh1ng), Scott Lundberg (@slundberg), Xu Xiao (@sperlingxx), @sriramch, Pasha Stetsenko (@st-pasha), Stephanie Yang (@stpyang), Yuan Tang (@terrytangyuan), Mathew Wicks (@thesuperzapper), Theodore Vasiloudis (@thvasilo), TinkleG (@tinkle1129), Oleksandr Pryimak (@trams), Jiaming Yuan (@trivialfis), Bobby Wang (@wbo4958), yage (@yage99), @yellowdolphin, Yin Lou (@yinlou)

## v0.90 (2019.05.18)

### XGBoost Python package drops Python 2.x (#4379, #4381)
Python 2.x is reaching its end-of-life at the end of this year. [Many scientific Python packages are now moving to drop Python 2.x](https://python3statement.org/).

### XGBoost4J-Spark now requires Spark 2.4.x (#4377)
* Spark 2.3 is reaching its end-of-life soon. See discussion at #4389.
* **Consistent handling of missing values** (#4309, #4349, #4411): Many users had reported issue with inconsistent predictions between XGBoost4J-Spark and the Python XGBoost package. The issue was caused by Spark mis-handling non-zero missing values (NaN, -1, 999 etc). We now alert the user whenever Spark doesn't handle missing values correctly (#4309, #4349). See [the tutorial for dealing with missing values in XGBoost4J-Spark](https://xgboost.readthedocs.io/en/release_0.90/jvm/xgboost4j_spark_tutorial.html#dealing-with-missing-values). This fix also depends on the availability of Spark 2.4.x.

### Roadmap: better performance scaling for multi-core CPUs (#4310)
* Poor performance scaling of the `hist` algorithm for multi-core CPUs has been under investigation (#3810). #4310 optimizes quantile sketches and other pre-processing tasks. Special thanks to @SmirnovEgorRu.

### Roadmap: Harden distributed training (#4250)
* Make distributed training in XGBoost more robust by hardening [Rabit](https://github.com/dmlc/rabit), which implements [the AllReduce primitive](https://en.wikipedia.org/wiki/Reduce_%28parallel_pattern%29). In particular, improve test coverage on mechanisms for fault tolerance and recovery. Special thanks to @chenqin.

### New feature: Multi-class metric functions for GPUs (#4368)
* Metrics for multi-class classification have been ported to GPU: `merror`, `mlogloss`. Special thanks to @trivialfis.
* With supported metrics, XGBoost will select the correct devices based on your system and `n_gpus` parameter.

### New feature: Scikit-learn-like random forest API (#4148, #4255, #4258)
* XGBoost Python package now offers `XGBRFClassifier` and `XGBRFRegressor` API to train random forests. See [the tutorial](https://xgboost.readthedocs.io/en/release_0.90/tutorials/rf.html). Special thanks to @canonizer

### New feature: use external memory in GPU predictor (#4284, #4396, #4438, #4457)
* It is now possible to make predictions on GPU when the input is read from external memory. This is useful when you want to make predictions with big dataset that does not fit into the GPU memory. Special thanks to @rongou, @canonizer, @sriramch.

  ```python
  dtest = xgboost.DMatrix('test_data.libsvm#dtest.cache')
  bst.set_param('predictor', 'gpu_predictor')
  bst.predict(dtest)
  ```

* Coming soon: GPU training (`gpu_hist`) with external memory

### New feature: XGBoost can now handle comments in LIBSVM files (#4430)
* Special thanks to @trivialfis and @hcho3

### New feature: Embed XGBoost in your C/C++ applications using CMake (#4323, #4333, #4453)
* It is now easier than ever to embed XGBoost in your C/C++ applications. In your CMakeLists.txt, add `xgboost::xgboost` as a linked library:

  ```cmake
  find_package(xgboost REQUIRED)
  add_executable(api-demo c-api-demo.c)
  target_link_libraries(api-demo xgboost::xgboost)
  ```

  [XGBoost C API documentation is available.](https://xgboost.readthedocs.io/en/release_0.90/dev) Special thanks to @trivialfis

### Performance improvements
* Use feature interaction constraints to narrow split search space (#4341, #4428)
* Additional optimizations for `gpu_hist` (#4248, #4283)
* Reduce OpenMP thread launches in `gpu_hist` (#4343)
* Additional optimizations for multi-node multi-GPU random forests. (#4238)
* Allocate unique prediction buffer for each input matrix, to avoid re-sizing GPU array (#4275)
* Remove various synchronisations from CUDA API calls (#4205)
* XGBoost4J-Spark
  - Allow the user to control whether to cache partitioned training data, to potentially reduce execution time (#4268)

### Bug-fixes
* Fix node reuse in `hist` (#4404)
* Fix GPU histogram allocation (#4347)
* Fix matrix attributes not sliced (#4311)
* Revise AUC and AUCPR metrics now work with weighted ranking task (#4216, #4436)
* Fix timer invocation for InitDataOnce() in `gpu_hist` (#4206)
* Fix R-devel errors (#4251)
* Make gradient update in GPU linear updater thread-safe (#4259)
* Prevent out-of-range access in column matrix (#4231)
* Don't store DMatrix handle in Python object until it's initialized, to improve exception safety (#4317)
* XGBoost4J-Spark
  - Fix non-deterministic order within a zipped partition on prediction (#4388)
  - Remove race condition on tracker shutdown (#4224)
  - Allow set the parameter `maxLeaves`. (#4226)
  - Allow partial evaluation of dataframe before prediction (#4407)
  - Automatically set `maximize_evaluation_metrics` if not explicitly given (#4446)

### API changes
* Deprecate `reg:linear` in favor of `reg:squarederror`. (#4267, #4427)
* Add attribute getter and setter to the Booster object in XGBoost4J (#4336)

### Maintenance: Refactor C++ code for legibility and maintainability
* Fix clang-tidy warnings. (#4149)
* Remove deprecated C APIs. (#4266)
* Use Monitor class to time functions in `hist`. (#4273)
* Retire DVec class in favour of c++20 style span for device memory. (#4293)
* Improve HostDeviceVector exception safety (#4301)

### Maintenance: testing, continuous integration, build system
* **Major refactor of CMakeLists.txt** (#4323, #4333, #4453): adopt modern CMake and export XGBoost as a target
* **Major improvement in Jenkins CI pipeline** (#4234)
  - Migrate all Linux tests to Jenkins (#4401)
  - Builds and tests are now de-coupled, to test an artifact against multiple versions of CUDA, JDK, and other dependencies (#4401)
  - Add Windows GPU to Jenkins CI pipeline (#4463, #4469)
* Support CUDA 10.1 (#4223, #4232, #4265, #4468)
* Python wheels are now built with CUDA 9.0, so that JIT is not required on Volta architecture (#4459)
* Integrate with NVTX CUDA profiler (#4205)
* Add a test for cpu predictor using external memory (#4308)
* Refactor tests to get rid of duplication (#4358)
* Remove test dependency on `craigcitro/r-travis`, since it's deprecated (#4353)
* Add files from local R build to `.gitignore` (#4346)
* Make XGBoost4J compatible with Java 9+ by revising NativeLibLoader (#4351)
* Jenkins build for CUDA 10.0 (#4281)
* Remove remaining `silent` and `debug_verbose` in Python tests (#4299)
* Use all cores to build XGBoost4J lib on linux (#4304)
* Upgrade Jenkins Linux build environment to GCC 5.3.1, CMake 3.6.0 (#4306)
* Make CMakeLists.txt compatible with CMake 3.3 (#4420)
* Add OpenMP option in CMakeLists.txt (#4339)
* Get rid of a few trivial compiler warnings (#4312)
* Add external Docker build cache, to speed up builds on Jenkins CI (#4331, #4334, #4458)
* Fix Windows tests (#4403)
* Fix a broken python test (#4395)
* Use a fixed seed to split data in XGBoost4J-Spark tests, for reproducibility (#4417)
* Add additional Python tests to test training under constraints (#4426)
* Enable building with shared NCCL. (#4447)

### Usability Improvements, Documentation
* Document limitation of one-split-at-a-time Greedy tree learning heuristic (#4233)
* Update build doc: PyPI wheel now support multi-GPU (#4219)
* Fix docs for `num_parallel_tree` (#4221)
* Fix document about `colsample_by*` parameter (#4340)
* Make the train and test input with same colnames. (#4329)
* Update R contribute link. (#4236)
* Fix travis R tests (#4277)
* Log version number in crash log in XGBoost4J-Spark (#4271, #4303)
* Allow supression of Rabit output in Booster::train in XGBoost4J (#4262)
* Add tutorial on handling missing values in XGBoost4J-Spark (#4425)
* Fix typos (#4345, #4393, #4432, #4435)
* Added language classifier in setup.py (#4327)
* Added Travis CI badge (#4344)
* Add BentoML to use case section (#4400)
* Remove subtly sexist remark (#4418)
* Add R vignette about parsing JSON dumps (#4439)

### Acknowledgement
**Contributors**: Nan Zhu (@CodingCat), Adam Pocock (@Craigacp), Daniel Hen (@Daniel8hen), Jiaxiang Li (@JiaxiangBU), Rory Mitchell (@RAMitchell), Egor Smirnov (@SmirnovEgorRu), Andy Adinets (@canonizer), Jonas (@elcombato), Harry Braviner (@harrybraviner), Philip Hyunsu Cho (@hcho3), Tong He (@hetong007), James Lamb (@jameslamb), Jean-Francois Zinque (@jeffzi), Yang Yang (@jokerkeny), Mayank Suman (@mayanksuman), jess (@monkeywithacupcake), Hajime Morrita (@omo), Ravi Kalia (@project-delphi), @ras44, Rong Ou (@rongou), Shaochen Shi (@shishaochen), Xu Xiao (@sperlingxx), @sriramch, Jiaming Yuan (@trivialfis), Christopher Suchanek (@wsuchy), Bozhao (@yubozhao)

**Reviewers**: Nan Zhu (@CodingCat), Adam Pocock (@Craigacp), Daniel Hen (@Daniel8hen), Jiaxiang Li (@JiaxiangBU), Laurae (@Laurae2), Rory Mitchell (@RAMitchell), Egor Smirnov (@SmirnovEgorRu), @alois-bissuel, Andy Adinets (@canonizer), Chen Qin (@chenqin), Harry Braviner (@harrybraviner), Philip Hyunsu Cho (@hcho3), Tong He (@hetong007), @jakirkham, James Lamb (@jameslamb), Julien Schueller (@jschueller), Mayank Suman (@mayanksuman), Hajime Morrita (@omo), Rong Ou (@rongou), Sara Robinson (@sararob), Shaochen Shi (@shishaochen), Xu Xiao (@sperlingxx), @sriramch, Sean Owen (@srowen), Sergei Lebedev (@superbobry), Yuan (Terry) Tang (@terrytangyuan), Theodore Vasiloudis (@thvasilo), Matthew Tovbin (@tovbinm), Jiaming Yuan (@trivialfis), Xin Yin (@xydrolase)

## v0.82 (2019.03.03)
This release is packed with many new features and bug fixes.

### Roadmap: better performance scaling for multi-core CPUs (#3957)
* Poor performance scaling of the `hist` algorithm for multi-core CPUs has been under investigation (#3810). #3957 marks an important step toward better performance scaling, by using software pre-fetching and replacing STL vectors with C-style arrays. Special thanks to @Laurae2 and @SmirnovEgorRu.
* See #3810 for latest progress on this roadmap.

### New feature: Distributed Fast Histogram Algorithm (`hist`) (#4011, #4102, #4140, #4128)
* It is now possible to run the `hist` algorithm in distributed setting. Special thanks to @CodingCat. The benefits include:
  1. Faster local computation via feature binning
  2. Support for monotonic constraints and feature interaction constraints
  3. Simpler codebase than `approx`, allowing for future improvement
* Depth-wise tree growing is now performed in a separate code path, so that cross-node syncronization is performed only once per level.

### New feature: Multi-Node, Multi-GPU training (#4095)
* Distributed training is now able to utilize clusters equipped with NVIDIA GPUs. In particular, the rabit AllReduce layer will communicate GPU device information. Special thanks to @mt-jones, @RAMitchell, @rongou, @trivialfis, @canonizer, and @jeffdk.
* Resource management systems will be able to assign a rank for each GPU in the cluster.
* In Dask, users will be able to construct a collection of XGBoost processes over an inhomogeneous device cluster (i.e. workers with different number and/or kinds of GPUs).

### New feature: Multiple validation datasets in XGBoost4J-Spark (#3904, #3910)
* You can now track the performance of the model during training with multiple evaluation datasets. By specifying `eval_sets` or call `setEvalSets` over a `XGBoostClassifier` or `XGBoostRegressor`, you can pass in multiple evaluation datasets typed as a `Map` from `String` to `DataFrame`. Special thanks to @CodingCat.
* See the usage of multiple validation datasets [here](https://github.com/dmlc/xgboost/blob/0c1d5f1120c0a159f2567b267f0ec4ffadee00d0/jvm-packages/xgboost4j-example/src/main/scala/ml/dmlc/xgboost4j/scala/example/spark/SparkTraining.scala#L66-L78)

### New feature: Additional metric functions for GPUs (#3952)
* Element-wise metrics have been ported to GPU: `rmse`, `mae`, `logloss`, `poisson-nloglik`, `gamma-deviance`, `gamma-nloglik`, `error`, `tweedie-nloglik`. Special thanks to @trivialfis and @RAMitchell.
* With supported metrics, XGBoost will select the correct devices based on your system and `n_gpus` parameter.

### New feature: Column sampling at individual nodes (splits) (#3971)
* Columns (features) can now be sampled at individual tree nodes, in addition to per-tree and per-level sampling. To enable per-node sampling, set `colsample_bynode` parameter, which represents the fraction of columns sampled at each node. This parameter is set to 1.0 by default (i.e. no sampling per node). Special thanks to @canonizer.
* The `colsample_bynode` parameter works cumulatively with other `colsample_by*` parameters: for example, `{'colsample_bynode':0.5, 'colsample_bytree':0.5}` with 100 columns will give 25 features to choose from at each split.

### Major API change: consistent logging level via `verbosity` (#3982, #4002, #4138)
* XGBoost now allows fine-grained control over logging. You can set `verbosity` to 0 (silent), 1 (warning), 2 (info), and 3 (debug). This is useful for controlling the amount of logging outputs. Special thanks to @trivialfis.
* Parameters `silent` and `debug_verbose` are now deprecated.
* Note: Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message.  If there's unexpected behaviour, please try to increase value of verbosity.

### Major bug fix: external memory (#4040, #4193)
* Clarify object ownership in multi-threaded prefetcher, to avoid memory error.
* Correctly merge two column batches (which uses [CSC layout](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))).
* Add unit tests for external memory.
* Special thanks to @trivialfis and @hcho3.

### Major bug fix: early stopping fixed in XGBoost4J and XGBoost4J-Spark (#3928, #4176)
* Early stopping in XGBoost4J and XGBoost4J-Spark is now consistent with its counterpart in the Python package. Training stops if the current iteration is `earlyStoppingSteps` away from the best iteration. If there are multiple evaluation sets, only the last one is used to determinate early stop.
* See the updated documentation [here](https://xgboost.readthedocs.io/en/release_0.82/jvm/xgboost4j_spark_tutorial.html#early-stopping)
* Special thanks to @CodingCat, @yanboliang, and @mingyang.

### Major bug fix: infrequent features should not crash distributed training (#4045)
* For infrequently occuring features, some partitions may not get any instance. This scenario used to crash distributed training due to mal-formed ranges. The problem has now been fixed.
* In practice, one-hot-encoded categorical variables tend to produce rare features, particularly when the cardinality is high.
* Special thanks to @CodingCat.

### Performance improvements
* Faster, more space-efficient radix sorting in `gpu_hist` (#3895)
* Subtraction trick in histogram calculation in `gpu_hist` (#3945)
* More performant re-partition in XGBoost4J-Spark (#4049)

### Bug-fixes
* Fix semantics of `gpu_id` when running multiple XGBoost processes on a multi-GPU machine (#3851)
* Fix page storage path for external memory on Windows (#3869)
* Fix configuration setup so that DART utilizes GPU (#4024)
* Eliminate NAN values from SHAP prediction (#3943)
* Prevent empty quantile sketches in `hist` (#4155)
* Enable running objectives with 0 GPU (#3878)
* Parameters are no longer dependent on system locale (#3891, #3907)
* Use consistent data type in the GPU coordinate descent code (#3917)
* Remove undefined behavior in the CLI config parser on the ARM platform (#3976)
* Initialize counters in GPU AllReduce (#3987)
* Prevent deadlocks in GPU AllReduce (#4113)
* Load correct values from sliced NumPy arrays (#4147, #4165)
* Fix incorrect GPU device selection (#4161)
* Make feature binning logic in `hist` aware of query groups when running a ranking task (#4115). For ranking task, query groups are weighted, not individual instances.
* Generate correct C++ exception type for `LOG(FATAL)` macro (#4159)
* Python package
  - Python package should run on system without `PATH` environment variable (#3845)
  - Fix `coef_` and `intercept_` signature to be compatible with `sklearn.RFECV` (#3873)
  - Use UTF-8 encoding in Python package README, to support non-English locale (#3867)
  - Add AUC-PR to list of metrics to maximize for early stopping (#3936)
  - Allow loading pickles without `self.booster` attribute, for backward compatibility (#3938, #3944)
  - White-list DART for feature importances (#4073)
  - Update usage of [h2oai/datatable](https://github.com/h2oai/datatable) (#4123)
* XGBoost4J-Spark
  - Address scalability issue in prediction (#4033)
  - Enforce the use of per-group weights for ranking task (#4118)
  - Fix vector size of `rawPredictionCol` in `XGBoostClassificationModel` (#3932)
  - More robust error handling in Spark tracker (#4046, #4108)
  - Fix return type of `setEvalSets` (#4105)
  - Return correct value of `getMaxLeaves` (#4114)

### API changes
* Add experimental parameter `single_precision_histogram` to use single-precision histograms for the `gpu_hist` algorithm (#3965)
* Python package
  - Add option to select type of feature importances in the scikit-learn inferface (#3876)
  - Add `trees_to_df()` method to dump decision trees as Pandas data frame (#4153)
  - Add options to control node shapes in the GraphViz plotting function (#3859)
  - Add `xgb_model` option to `XGBClassifier`, to load previously saved model (#4092)
  - Passing lists into `DMatrix` is now deprecated (#3970)
* XGBoost4J
  - Support multiple feature importance features (#3801)

### Maintenance: Refactor C++ code for legibility and maintainability
* Refactor `hist` algorithm code and add unit tests (#3836)
* Minor refactoring of split evaluator in `gpu_hist` (#3889)
* Removed unused leaf vector field in the tree model (#3989)
* Simplify the tree representation by combining `TreeModel` and `RegTree` classes (#3995)
* Simplify and harden tree expansion code (#4008, #4015)
* De-duplicate parameter classes in the linear model algorithms (#4013)
* Robust handling of ranges with C++20 span in `gpu_exact` and `gpu_coord_descent` (#4020, #4029)
* Simplify tree training code (#3825). Also use Span class for robust handling of ranges.

### Maintenance: testing, continuous integration, build system
* Disallow `std::regex` since it's not supported by GCC 4.8.x (#3870)
* Add multi-GPU tests for coordinate descent algorithm for linear models (#3893, #3974)
* Enforce naming style in Python lint (#3896)
* Refactor Python tests (#3897, #3901): Use pytest exclusively, display full trace upon failure
* Address `DeprecationWarning` when using Python collections (#3909)
* Use correct group for maven site plugin (#3937)
* Jenkins CI is now using on-demand EC2 instances exclusively, due to unreliability of Spot instances (#3948)
* Better GPU performance logging (#3945)
* Fix GPU tests on machines with only 1 GPU (#4053)
* Eliminate CRAN check warnings and notes (#3988)
* Add unit tests for tree serialization (#3989)
* Add unit tests for tree fitting functions in `hist` (#4155)
* Add a unit test for `gpu_exact` algorithm (#4020)
* Correct JVM CMake GPU flag (#4071)
* Fix failing Travis CI on Mac (#4086)
* Speed up Jenkins by not compiling CMake (#4099)
* Analyze C++ and CUDA code using clang-tidy, as part of Jenkins CI pipeline (#4034)
* Fix broken R test: Install Homebrew GCC (#4142)
* Check for empty datasets in GPU unit tests (#4151)
* Fix Windows compilation (#4139)
* Comply with latest convention of cpplint (#4157)
* Fix a unit test in `gpu_hist` (#4158)
* Speed up data generation in Python tests (#4164)

### Usability Improvements
* Add link to [InfoWorld 2019 Technology of the Year Award](https://www.infoworld.com/article/3336072/application-development/infoworlds-2019-technology-of-the-year-award-winners.html) (#4116)
* Remove outdated AWS YARN tutorial (#3885)
* Document current limitation in number of features (#3886)
* Remove unnecessary warning when `gblinear` is selected (#3888)
* Document limitation of CSV parser: header not supported (#3934)
* Log training parameters in XGBoost4J-Spark (#4091)
* Clarify early stopping behavior in the scikit-learn interface (#3967)
* Clarify behavior of `max_depth` parameter (#4078)
* Revise Python docstrings for ranking task (#4121). In particular, weights must be per-group in learning-to-rank setting.
* Document parameter `num_parallel_tree` (#4022)
* Add Jenkins status badge (#4090)
* Warn users against using internal functions of `Booster` object (#4066)
* Reformat `benchmark_tree.py` to comply with Python style convention (#4126)
* Clarify a comment in `objectiveTrait` (#4174)
* Fix typos and broken links in documentation (#3890, #3872, #3902, #3919, #3975, #4027, #4156, #4167)

### Acknowledgement
**Contributors** (in no particular order): Jiaming Yuan (@trivialfis), Hyunsu Cho (@hcho3), Nan Zhu (@CodingCat), Rory Mitchell (@RAMitchell), Yanbo Liang (@yanboliang), Andy Adinets (@canonizer), Tong He (@hetong007), Yuan Tang (@terrytangyuan)

**First-time Contributors** (in no particular order): Jelle Zijlstra (@JelleZijlstra), Jiacheng Xu (@jiachengxu), @ajing, Kashif Rasul (@kashif), @theycallhimavi, Joey Gao (@pjgao), Prabakaran Kumaresshan (@nixphix), Huafeng Wang (@huafengw), @lyxthe, Sam Wilkinson (@scwilkinson), Tatsuhito Kato (@stabacov), Shayak Banerjee (@shayakbanerjee), Kodi Arfer (@Kodiologist), @KyleLi1985, Egor Smirnov (@SmirnovEgorRu), @tmitanitky, Pasha Stetsenko (@st-pasha), Kenichi Nagahara (@keni-chi), Abhai Kollara Dilip (@abhaikollara), Patrick Ford (@pford221), @hshujuan, Matthew Jones (@mt-jones), Thejaswi Rao (@teju85), Adam November (@anovember)

**First-time Reviewers** (in no particular order): Mingyang Hu (@mingyang), Theodore Vasiloudis (@thvasilo), Jakub Troszok (@troszok), Rong Ou (@rongou), @Denisevi4, Matthew Jones (@mt-jones), Jeff Kaplan (@jeffdk)

## v0.81 (2018.11.04)
### New feature: feature interaction constraints
* Users are now able to control which features (independent variables) are allowed to interact by specifying feature interaction constraints (#3466).
* [Tutorial](https://xgboost.readthedocs.io/en/release_0.81/tutorials/feature_interaction_constraint.html) is available, as well as [R](https://github.com/dmlc/xgboost/blob/9254c58e4dfff6a59dc0829a2ceb02e45ed17cd0/R-package/demo/interaction_constraints.R) and [Python](https://github.com/dmlc/xgboost/blob/9254c58e4dfff6a59dc0829a2ceb02e45ed17cd0/tests/python/test_interaction_constraints.py) examples.

### New feature: learning to rank using scikit-learn interface
* Learning to rank task is now available for the scikit-learn interface of the Python package (#3560, #3848). It is now possible to integrate the XGBoost ranking model into the scikit-learn learning pipeline.
* Examples of using `XGBRanker` class is found at [demo/rank/rank_sklearn.py](https://github.com/dmlc/xgboost/blob/24a268a2e3cb17302db3d72da8f04016b7d352d9/demo/rank/rank_sklearn.py).

### New feature: R interface for SHAP interactions
* SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model. Previously, this feature was only available from the Python package; now it is available from the R package as well (#3636).

### New feature: GPU predictor now use multiple GPUs to predict
* GPU predictor is now able to utilize multiple GPUs at once to accelerate prediction (#3738)

### New feature: Scale distributed XGBoost to large-scale clusters
* Fix OS file descriptor limit assertion error on large cluster (#3835, dmlc/rabit#73) by replacing `select()` based AllReduce/Broadcast with `poll()` based implementation.
* Mitigate tracker "thundering herd" issue on large cluster. Add exponential backoff retry when workers connect to tracker.
* With this change, we were able to scale to 1.5k executors on a 12 billion row dataset after some tweaks here and there.

### New feature: Additional objective functions for GPUs
* New objective functions ported to GPU: `hinge`, `multi:softmax`, `multi:softprob`, `count:poisson`, `reg:gamma`, `"reg:tweedie`.
* With supported objectives, XGBoost will select the correct devices based on your system and `n_gpus` parameter.

### Major bug fix: learning to rank with XGBoost4J-Spark
* Previously, `repartitionForData` would shuffle data and lose ordering necessary for ranking task.
* To fix this issue, data points within each RDD partition is explicitly group by their group (query session) IDs (#3654). Also handle empty RDD partition carefully (#3750).

### Major bug fix: early stopping fixed in XGBoost4J-Spark
* Earlier implementation of early stopping had incorrect semantics and didn't let users to specify direction for optimizing (maximize / minimize)
* A parameter `maximize_evaluation_metrics` is defined so as to tell whether a metric should be maximized or minimized as part of early stopping criteria (#3808). Also early stopping now has correct semantics.

### API changes
* Column sampling by level (`colsample_bylevel`) is now functional for `hist` algorithm (#3635, #3862)
* GPU tag `gpu:` for regression objectives are now deprecated. XGBoost will select the correct devices automatically (#3643)
* Add `disable_default_eval_metric` parameter to disable default metric (#3606)
* Experimental AVX support for gradient computation is removed (#3752)
* XGBoost4J-Spark
  - Add `rank:ndcg` and `rank:map` to supported objectives (#3697)
* Python package
  - Add `callbacks` argument to `fit()` function of sciki-learn API (#3682)
  - Add `XGBRanker` to scikit-learn interface (#3560, #3848)
  - Add `validate_features` argument to `predict()` function of scikit-learn API (#3653)
  - Allow scikit-learn grid search over parameters specified as keyword arguments (#3791)
  - Add `coef_` and `intercept_` as properties of scikit-learn wrapper (#3855). Some scikit-learn functions expect these properties.

### Performance improvements
* Address very high GPU memory usage for large data (#3635)
* Fix performance regression within `EvaluateSplits()` of `gpu_hist` algorithm. (#3680)

### Bug-fixes
* Fix a problem in GPU quantile sketch with tiny instance weights. (#3628)
* Fix copy constructor for `HostDeviceVectorImpl` to prevent dangling pointers (#3657)
* Fix a bug in partitioned file loading (#3673)
* Fixed an uninitialized pointer in `gpu_hist` (#3703)
* Reshared data among GPUs when number of GPUs is changed (#3721)
* Add back `max_delta_step` to split evaluation (#3668)
* Do not round up integer thresholds for integer features in JSON dump (#3717)
* Use `dmlc::TemporaryDirectory` to handle temporaries in cross-platform way (#3783)
* Fix accuracy problem with `gpu_hist` when `min_child_weight` and `lambda` are set to 0 (#3793)
* Make sure that `tree_method` parameter is recognized and not silently ignored (#3849)
* XGBoost4J-Spark
  - Make sure `thresholds` are considered when executing `predict()` method (#3577)
  - Avoid losing precision when computing probabilities by converting to `Double` early (#3576)
  - `getTreeLimit()` should return `Int` (#3602)
  - Fix checkpoint serialization on HDFS (#3614)
  - Throw `ControlThrowable` instead of `InterruptedException` so that it is properly re-thrown (#3632)
  - Remove extraneous output to stdout (#3665)
  - Allow specification of task type for custom objectives and evaluations (#3646)
  - Fix distributed updater check (#3739)
  - Fix issue when spark job execution thread cannot return before we execute `first()` (#3758)
* Python package
  - Fix accessing `DMatrix.handle` before it is set (#3599)
  - `XGBClassifier.predict()` should return margin scores when `output_margin` is set to true (#3651)
  - Early stopping callback should maximize metric of form `NDCG@n-` (#3685)
  - Preserve feature names when slicing `DMatrix` (#3766)
* R package
  - Replace `nround` with `nrounds` to match actual parameter (#3592)
  - Amend `xgb.createFolds` to handle classes of a single element (#3630)
  - Fix buggy random generator and make `colsample_bytree` functional (#3781)

### Maintenance: testing, continuous integration, build system
* Add sanitizers tests to Travis CI (#3557)
* Add NumPy, Matplotlib, Graphviz as requirements for doc build (#3669)
* Comply with CRAN submission policy (#3660, #3728)
* Remove copy-paste error in JVM test suite (#3692)
* Disable flaky tests in `R-package/tests/testthat/test_update.R` (#3723)
* Make Python tests compatible with scikit-learn 0.20 release (#3731)
* Separate out restricted and unrestricted tasks, so that pull requests don't build downloadable artifacts (#3736)
* Add multi-GPU unit test environment (#3741)
* Allow plug-ins to be built by CMake (#3752)
* Test wheel compatibility on CPU containers for pull requests (#3762)
* Fix broken doc build due to Matplotlib 3.0 release (#3764)
* Produce `xgboost.so` for XGBoost-R on Mac OSX, so that `make install` works (#3767)
* Retry Jenkins CI tests up to 3 times to improve reliability (#3769, #3769, #3775, #3776, #3777)
* Add basic unit tests for `gpu_hist` algorithm (#3785)
* Fix Python environment for distributed unit tests (#3806)
* Test wheels on CUDA 10.0 container for compatibility (#3838)
* Fix JVM doc build (#3853)

### Maintenance: Refactor C++ code for legibility and maintainability
* Merge generic device helper functions into `GPUSet` class (#3626)
* Re-factor column sampling logic into `ColumnSampler` class (#3635, #3637)
* Replace `std::vector` with `HostDeviceVector` in `MetaInfo` and `SparsePage` (#3446)
* Simplify `DMatrix` class (#3395)
* De-duplicate CPU/GPU code using `Transform` class (#3643, #3751)
* Remove obsoleted `QuantileHistMaker` class (#3761)
* Remove obsoleted `NoConstraint` class (#3792)

### Other Features
* C++20-compliant Span class for safe pointer indexing (#3548, #3588)
* Add helper functions to manipulate multiple GPU devices (#3693)
* XGBoost4J-Spark
  - Allow specifying host ip from the `xgboost-tracker.properties file` (#3833). This comes in handy when `hosts` files doesn't correctly define localhost.

### Usability Improvements
* Add reference to GitHub repository in `pom.xml` of JVM packages (#3589)
* Add R demo of multi-class classification (#3695)
* Document JSON dump functionality (#3600, #3603)
* Document CUDA requirement and lack of external memory for GPU algorithms (#3624)
* Document LambdaMART objectives, both pairwise and listwise (#3672)
* Document `aucpr` evaluation metric (#3687)
* Document gblinear parameters: `feature_selector` and `top_k` (#3780)
* Add instructions for using MinGW-built XGBoost with Python. (#3774)
* Removed nonexistent parameter `use_buffer` from documentation (#3610)
* Update Python API doc to include all classes and members (#3619, #3682)
* Fix typos and broken links in documentation (#3618, #3640, #3676, #3713, #3759, #3784, #3843, #3852)
* Binary classification demo should produce LIBSVM with 0-based indexing (#3652)
* Process data once for Python and CLI examples of learning to rank (#3666)
* Include full text of Apache 2.0 license in the repository (#3698)
* Save predictor parameters in model file (#3856)
* JVM packages
  - Let users specify feature names when calling `getModelDump` and `getFeatureScore` (#3733)
  - Warn the user about the lack of over-the-wire encryption (#3667)
  - Fix errors in examples (#3719)
  - Document choice of trackers (#3831)
  - Document that vanilla Apache Spark is required (#3854)
* Python package
  - Document that custom objective can't contain colon (:) (#3601)
  - Show a better error message for failed library loading (#3690)
  - Document that feature importance is unavailable for non-tree learners (#3765)
  - Document behavior of `get_fscore()` for zero-importance features (#3763)
  - Recommend pickling as the way to save `XGBClassifier` / `XGBRegressor` / `XGBRanker` (#3829)
* R package
  - Enlarge variable importance plot to make it more visible (#3820)

### BREAKING CHANGES
* External memory page files have changed, breaking backwards compatibility for temporary storage used during external memory training. This only affects external memory users upgrading their xgboost version - we recommend clearing all `*.page` files before resuming training. Model serialization is unaffected.

### Known issues
* Quantile sketcher fails to produce any quantile for some edge cases (#2943)
* The `hist` algorithm leaks memory when used with learning rate decay callback (#3579)
* Using custom evaluation function together with early stopping causes assertion failure in XGBoost4J-Spark (#3595)
* Early stopping doesn't work with `gblinear` learner (#3789)
* Label and weight vectors are not reshared upon the change in number of GPUs (#3794). To get around this issue, delete the `DMatrix` object and re-load.
* The `DMatrix` Python objects are initialized with incorrect values when given array slices (#3841)
* The `gpu_id` parameter is broken and not yet properly supported (#3850)

### Acknowledgement
**Contributors** (in no particular order): Hyunsu Cho (@hcho3), Jiaming Yuan (@trivialfis), Nan Zhu (@CodingCat), Rory Mitchell (@RAMitchell), Andy Adinets (@canonizer), Vadim Khotilovich (@khotilov), Sergei Lebedev (@superbobry)

**First-time Contributors** (in no particular order): Matthew Tovbin (@tovbinm), Jakob Richter (@jakob-r), Grace Lam (@grace-lam), Grant W Schneider (@grantschneider), Andrew Thia (@BlueTea88), Sergei Chipiga (@schipiga), Joseph Bradley (@jkbradley), Chen Qin (@chenqin), Jerry Lin (@linjer), Dmitriy Rybalko (@rdtft), Michael Mui (@mmui), Takahiro Kojima (@515hikaru), Bruce Zhao (@BruceZhaoR), Wei Tian (@weitian), Saumya Bhatnagar (@Sam1301), Juzer Shakir (@JuzerShakir), Zhao Hang (@cleghom), Jonathan Friedman (@jontonsoup), Bruno Tremblay (@meztez), Boris Filippov (@frenzykryger), @Shiki-H, @mrgutkun, @gorogm, @htgeis, @jakehoare, @zengxy, @KOLANICH

**First-time Reviewers** (in no particular order): Nikita Titov (@StrikerRUS), Xiangrui Meng (@mengxr), Nirmal Borah (@Nirmal-Neel)


## v0.80 (2018.08.13)
* **JVM packages received a major upgrade**: To consolidate the APIs and improve the user experience, we refactored the design of XGBoost4J-Spark in a significant manner. (#3387)
  - Consolidated APIs: It is now much easier to integrate XGBoost models into a Spark ML pipeline. Users can control behaviors like output leaf prediction results by setting corresponding column names. Training is now more consistent with other Estimators in Spark MLLIB: there is now one single method `fit()` to train decision trees.
  - Better user experience: we refactored the parameters relevant modules in XGBoost4J-Spark to provide both camel-case (Spark ML style) and underscore (XGBoost style) parameters
  - A brand-new tutorial is [available](https://xgboost.readthedocs.io/en/release_0.80/jvm/xgboost4j_spark_tutorial.html) for XGBoost4J-Spark.
  - Latest API documentation is now hosted at https://xgboost.readthedocs.io/.
* XGBoost documentation now keeps track of multiple versions:
  - Latest master: https://xgboost.readthedocs.io/en/latest
  - 0.80 stable: https://xgboost.readthedocs.io/en/release_0.80
  - 0.72 stable: https://xgboost.readthedocs.io/en/release_0.72
* Support for per-group weights in ranking objective (#3379)
* Fix inaccurate decimal parsing (#3546)
* New functionality
  - Query ID column support in LIBSVM data files (#2749). This is convenient for performing ranking task in distributed setting.
  - Hinge loss for binary classification (`binary:hinge`) (#3477)
  - Ability to specify delimiter and instance weight column for CSV files (#3546)
  - Ability to use 1-based indexing instead of 0-based (#3546)
* GPU support
  - Quantile sketch, binning, and index compression are now performed on GPU, eliminating PCIe transfer for 'gpu_hist' algorithm (#3319, #3393)
  - Upgrade to NCCL2 for multi-GPU training (#3404).
  - Use shared memory atomics for faster training (#3384).
  - Dynamically allocate GPU memory, to prevent large allocations for deep trees (#3519)
  - Fix memory copy bug for large files (#3472)
* Python package
  - Importing data from Python datatable (#3272)
  - Pre-built binary wheels available for 64-bit Linux and Windows (#3424, #3443)
  - Add new importance measures 'total_gain', 'total_cover' (#3498)
  - Sklearn API now supports saving and loading models (#3192)
  - Arbitrary cross validation fold indices (#3353)
  - `predict()` function in Sklearn API uses `best_ntree_limit` if available, to make early stopping easier to use (#3445)
  - Informational messages are now directed to Python's `print()` rather than standard output (#3438). This way, messages appear inside Jupyter notebooks.
* R package
  - Oracle Solaris support, per CRAN policy (#3372)
* JVM packages
  - Single-instance prediction (#3464)
  - Pre-built JARs are now available from Maven Central (#3401)
  - Add NULL pointer check (#3021)
  - Consider `spark.task.cpus` when controlling parallelism (#3530)
  - Handle missing values in prediction (#3529)
  - Eliminate outputs of `System.out` (#3572)
* Refactored C++ DMatrix class for simplicity and de-duplication (#3301)
* Refactored C++ histogram facilities (#3564)
* Refactored constraints / regularization mechanism for split finding (#3335, #3429). Users may specify an elastic net (L2 + L1 regularization) on leaf weights as well as monotonic constraints on test nodes. The refactor will be useful for a future addition of feature interaction constraints.
* Statically link `libstdc++` for MinGW32 (#3430)
* Enable loading from `group`, `base_margin` and `weight` (see [here](http://xgboost.readthedocs.io/en/latest/tutorials/input_format.html#auxiliary-files-for-additional-information)) for Python, R, and JVM packages (#3431)
* Fix model saving for `count:possion` so that `max_delta_step` doesn't get truncated (#3515)
* Fix loading of sparse CSC matrix (#3553)
* Fix incorrect handling of `base_score` parameter for Tweedie regression (#3295)

## v0.72.1 (2018.07.08)
This version is only applicable for the Python package. The content is identical to that of v0.72.

## v0.72 (2018.06.01)
* Starting with this release, we plan to make a new release every two months. See #3252 for more details.
* Fix a pathological behavior (near-zero second-order gradients) in multiclass objective (#3304)
* Tree dumps now use high precision in storing floating-point values (#3298)
* Submodules `rabit` and `dmlc-core` have been brought up to date, bringing bug fixes (#3330, #3221).
* GPU support
  - Continuous integration tests for GPU code (#3294, #3309)
  - GPU accelerated coordinate descent algorithm (#3178)
  - Abstract 1D vector class now works with multiple GPUs (#3287)
  - Generate PTX code for most recent architecture (#3316)
  - Fix a memory bug on NVIDIA K80 cards (#3293)
  - Address performance instability for single-GPU, multi-core machines (#3324)
* Python package
  - FreeBSD support (#3247)
  - Validation of feature names in `Booster.predict()` is now optional (#3323)
* Updated Sklearn API
  - Validation sets now support instance weights (#2354)
  - `XGBClassifier.predict_proba()` should not support `output_margin` option. (#3343) See BREAKING CHANGES below.
* R package:
  - Better handling of NULL in `print.xgb.Booster()` (#3338)
  - Comply with CRAN policy by removing compiler warning suppression (#3329)
  - Updated CRAN submission
* JVM packages
  - JVM packages will now use the same versioning scheme as other packages (#3253)
  - Update Spark to 2.3 (#3254)
  - Add scripts to cross-build and deploy artifacts (#3276, #3307)
  - Fix a compilation error for Scala 2.10 (#3332)
* BREAKING CHANGES
  - `XGBClassifier.predict_proba()` no longer accepts parameter `output_margin`. The parameter makes no sense for `predict_proba()` because the method is to predict class probabilities, not raw margin scores.

## v0.71 (2018.04.11)
* This is a minor release, mainly motivated by issues concerning `pip install`, e.g. #2426, #3189, #3118, and #3194.
  With this release, users of Linux and MacOS will be able to run `pip install` for the most part.
* Refactored linear booster class (`gblinear`), so as to support multiple coordinate descent updaters (#3103, #3134). See BREAKING CHANGES below.
* Fix slow training for multiclass classification with high number of classes (#3109)
* Fix a corner case in approximate quantile sketch (#3167). Applicable for 'hist' and 'gpu_hist' algorithms
* Fix memory leak in DMatrix (#3182)
* New functionality
  - Better linear booster class (#3103, #3134)
  - Pairwise SHAP interaction effects (#3043)
  - Cox loss (#3043)
  - AUC-PR metric for ranking task (#3172)
  - Monotonic constraints for 'hist' algorithm (#3085)
* GPU support
    - Create an abstract 1D vector class that moves data seamlessly between the main and GPU memory (#2935, #3116, #3068). This eliminates unnecessary PCIe data transfer during training time.
  - Fix minor bugs (#3051, #3217)
  - Fix compatibility error for CUDA 9.1 (#3218)
* Python package:
  - Correctly handle parameter `verbose_eval=0` (#3115)
* R package:
  - Eliminate segmentation fault on 32-bit Windows platform (#2994)
* JVM packages
  - Fix a memory bug involving double-freeing Booster objects (#3005, #3011)
  - Handle empty partition in predict (#3014)
  - Update docs and unify terminology (#3024)
  - Delete cache files after job finishes (#3022)
  - Compatibility fixes for latest Spark versions (#3062, #3093)
* BREAKING CHANGES: Updated linear modelling algorithms. In particular L1/L2 regularisation penalties are now normalised to number of training examples. This makes the implementation consistent with sklearn/glmnet. L2 regularisation has also been removed from the intercept. To produce linear models with the old regularisation behaviour, the alpha/lambda regularisation parameters can be manually scaled by dividing them by the number of training examples.

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
* Faster construction of `DMatrix` from 2D NumPy matrices: elide copies, use of multiple threads
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
  - Faster gradient calculation using AVX SIMD
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
