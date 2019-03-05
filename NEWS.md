XGBoost Change Log
==================

This file records the changes in xgboost library in reverse chronological order.

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
* Using custom evaluation funciton together with early stopping causes assertion failure in XGBoost4J-Spark (#3595)
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
  - `XGBClassifier.predict_proba()` no longer accepts paramter `output_margin`. The paramater makes no sense for `predict_proba()` because the method is to predict class probabilities, not raw margin scores.

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
  - Create an abtract 1D vector class that moves data seamlessly between the main and GPU memory (#2935, #3116, #3068). This eliminates unnecessary PCIe data transfer during training time.
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
