XGBoost Change Log
==================

This file records the changes in xgboost library in reverse chronological order.

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
* Remove plugin, cuda related code in automake & autoconf files (#4789)
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
* Remove doc about not supporting cuda 10.1 (#4578)
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
* Monitor for distributed envorinment (#4829). This is useful for identifying performance bottleneck.
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
