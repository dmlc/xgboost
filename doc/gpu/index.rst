###################
XGBoost GPU Support
###################

This page contains information about GPU algorithms supported in XGBoost.
To install GPU support, checkout the :doc:`/build`.

.. note:: CUDA 10.0, Compute Capability 3.5 required

  The GPU algorithms in XGBoost require a graphics card with compute capability 3.5 or higher, with
  CUDA toolkits 10.0 or later.
  (See `this list <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ to look up compute capability of your GPU card.)

*********************************************
CUDA Accelerated Tree Construction Algorithms
*********************************************
Tree construction (training) and prediction can be accelerated with CUDA-capable GPUs.

Usage
=====
Specify the ``tree_method`` parameter as one of the following algorithms.

Algorithms
----------

+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tree_method           | Description                                                                                                                                                           |
+=======================+=======================================================================================================================================================================+
| gpu_hist              | Equivalent to the XGBoost fast histogram algorithm. Much faster and uses considerably less memory. NOTE: May run very slowly on GPUs older than Pascal architecture.  |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Supported parameters
--------------------

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+--------------------------------+--------------+
| parameter                      | ``gpu_hist`` |
+================================+==============+
| ``subsample``                  | |tick|       |
+--------------------------------+--------------+
| ``sampling_method``            | |tick|       |
+--------------------------------+--------------+
| ``colsample_bytree``           | |tick|       |
+--------------------------------+--------------+
| ``colsample_bylevel``          | |tick|       |
+--------------------------------+--------------+
| ``max_bin``                    | |tick|       |
+--------------------------------+--------------+
| ``gamma``                      | |tick|       |
+--------------------------------+--------------+
| ``gpu_id``                     | |tick|       |
+--------------------------------+--------------+
| ``predictor``                  | |tick|       |
+--------------------------------+--------------+
| ``grow_policy``                | |tick|       |
+--------------------------------+--------------+
| ``monotone_constraints``       | |tick|       |
+--------------------------------+--------------+
| ``interaction_constraints``    | |tick|       |
+--------------------------------+--------------+
| ``single_precision_histogram`` | |tick|       |
+--------------------------------+--------------+

GPU accelerated prediction is enabled by default for the above mentioned ``tree_method`` parameters but can be switched to CPU prediction by setting ``predictor`` to ``cpu_predictor``. This could be useful if you want to conserve GPU memory. Likewise when using CPU algorithms, GPU accelerated prediction can be enabled by setting ``predictor`` to ``gpu_predictor``.

The experimental parameter ``single_precision_histogram`` can be set to True to enable building histograms using single precision. This may improve speed, in particular on older architectures.

The device ordinal (which GPU to use if you have many of them) can be selected using the
``gpu_id`` parameter, which defaults to 0 (the first device reported by CUDA runtime).


The GPU algorithms currently work with CLI, Python and R packages. See :doc:`/build` for details.

.. code-block:: python
  :caption: Python example

  param['gpu_id'] = 0
  param['tree_method'] = 'gpu_hist'

.. code-block:: python
  :caption: With Scikit-Learn interface

  XGBRegressor(tree_method='gpu_hist', gpu_id=0)


GPU-Accelerated SHAP values
=============================
XGBoost makes use of `GPUTreeShap <https://github.com/rapidsai/gputreeshap>`_ as a backend for computing shap values when the GPU predictor is selected.

.. code-block:: python

  model.set_param({"predictor": "gpu_predictor"})
  shap_values = model.predict(dtrain, pred_contribs=True)
  shap_interaction_values = model.predict(dtrain, pred_interactions=True)

See examples `here
<https://github.com/dmlc/xgboost/tree/master/demo/gpu_acceleration>`_.

Multi-node Multi-GPU Training
=============================
XGBoost supports fully distributed GPU training using `Dask <https://dask.org/>`_. For
getting started see our tutorial :doc:`/tutorials/dask` and worked examples `here
<https://github.com/dmlc/xgboost/tree/master/demo/dask>`_, also Python documentation
:ref:`dask_api` for complete reference.


Objective functions
===================
Most of the objective functions implemented in XGBoost can be run on GPU.  Following table shows current support status.

+----------------------+-------------+
| Objectives           | GPU support |
+----------------------+-------------+
| reg:squarederror     | |tick|      |
+----------------------+-------------+
| reg:squaredlogerror  | |tick|      |
+----------------------+-------------+
| reg:logistic         | |tick|      |
+----------------------+-------------+
| reg:pseudohubererror | |tick|      |
+----------------------+-------------+
| binary:logistic      | |tick|      |
+----------------------+-------------+
| binary:logitraw      | |tick|      |
+----------------------+-------------+
| binary:hinge         | |tick|      |
+----------------------+-------------+
| count:poisson        | |tick|      |
+----------------------+-------------+
| reg:gamma            | |tick|      |
+----------------------+-------------+
| reg:tweedie          | |tick|      |
+----------------------+-------------+
| multi:softmax        | |tick|      |
+----------------------+-------------+
| multi:softprob       | |tick|      |
+----------------------+-------------+
| survival:cox         | |cross|     |
+----------------------+-------------+
| survival:aft         | |tick|      |
+----------------------+-------------+
| rank:pairwise        | |tick|      |
+----------------------+-------------+
| rank:ndcg            | |tick|      |
+----------------------+-------------+
| rank:map             | |tick|      |
+----------------------+-------------+

Objective will run on GPU if GPU updater (``gpu_hist``), otherwise they will run on CPU by
default.  For unsupported objectives XGBoost will fall back to using CPU implementation by
default.  Note that when using GPU ranking objective, the result is not deterministic due
to the non-associative aspect of floating point summation.

Metric functions
===================
Following table shows current support status for evaluation metrics on the GPU.

+------------------------------+-------------+
| Metric                       | GPU Support |
+==============================+=============+
| rmse                         | |tick|      |
+------------------------------+-------------+
| rmsle                        | |tick|      |
+------------------------------+-------------+
| mae                          | |tick|      |
+------------------------------+-------------+
| mape                         | |tick|      |
+------------------------------+-------------+
| mphe                         | |tick|      |
+------------------------------+-------------+
| logloss                      | |tick|      |
+------------------------------+-------------+
| error                        | |tick|      |
+------------------------------+-------------+
| merror                       | |tick|      |
+------------------------------+-------------+
| mlogloss                     | |tick|      |
+------------------------------+-------------+
| auc                          | |tick|      |
+------------------------------+-------------+
| aucpr                        | |cross|     |
+------------------------------+-------------+
| ndcg                         | |tick|      |
+------------------------------+-------------+
| map                          | |tick|      |
+------------------------------+-------------+
| poisson-nloglik              | |tick|      |
+------------------------------+-------------+
| gamma-nloglik                | |tick|      |
+------------------------------+-------------+
| cox-nloglik                  | |cross|     |
+------------------------------+-------------+
| aft-nloglik                  | |tick|      |
+------------------------------+-------------+
| interval-regression-accuracy | |tick|      |
+------------------------------+-------------+
| gamma-deviance               | |tick|      |
+------------------------------+-------------+
| tweedie-nloglik              | |tick|      |
+------------------------------+-------------+

Similar to objective functions, default device for metrics is selected based on tree
updater and predictor (which is selected based on tree updater).

Benchmarks
==========
You can run benchmarks on synthetic data for binary classification:

.. code-block:: bash

  python tests/benchmark/benchmark_tree.py --tree_method=gpu_hist
  python tests/benchmark/benchmark_tree.py --tree_method=hist

Training time on 1,000,000 rows x 50 columns of random data with 500 boosting iterations and 0.25/0.75 test/train split with AMD Ryzen 7 2700 8 core @3.20GHz and Nvidia 1080ti yields the following results:

+--------------+----------+
| tree_method  | Time (s) |
+==============+==========+
| gpu_hist     | 12.57    |
+--------------+----------+
| hist         | 36.01    |
+--------------+----------+

Memory usage
============
The following are some guidelines on the device memory usage of the `gpu_hist` updater.

If you train xgboost in a loop you may notice xgboost is not freeing device memory after each training iteration. This is because memory is allocated over the lifetime of the booster object and does not get freed until the booster is freed. A workaround is to serialise the booster object after training. See `demo/gpu_acceleration/memory.py` for a simple example.

Memory inside xgboost training is generally allocated for two reasons - storing the dataset and working memory.

The dataset itself is stored on device in a compressed ELLPACK format. The ELLPACK format is a type of sparse matrix that stores elements with a constant row stride. This format is convenient for parallel computation when compared to CSR because the row index of each element is known directly from its address in memory. The disadvantage of the ELLPACK format is that it becomes less memory efficient if the maximum row length is significantly more than the average row length. Elements are quantised and stored as integers. These integers are compressed to a minimum bit length. Depending on the number of features, we usually don't need the full range of a 32 bit integer to store elements and so compress this down. The compressed, quantised ELLPACK format will commonly use 1/4 the space of a CSR matrix stored in floating point.

In some cases the full CSR matrix stored in floating point needs to be allocated on the device. This currently occurs for prediction in multiclass classification. If this is a problem consider setting `'predictor'='cpu_predictor'`. This also occurs when the external data itself comes from a source on device e.g. a cudf DataFrame. These are known issues we hope to resolve.

Working memory is allocated inside the algorithm proportional to the number of rows to keep track of gradients, tree positions and other per row statistics. Memory is allocated for histogram bins proportional to the number of bins, number of features and nodes in the tree. For performance reasons we keep histograms in memory from previous nodes in the tree, when a certain threshold of memory usage is passed we stop doing this to conserve memory at some performance loss.

The quantile finding algorithm also uses some amount of working device memory. It is able to operate in batches, but is not currently well optimised for sparse data.

If you are getting out-of-memory errors on a big dataset, try the `external memory version <../tutorials/external_memory.html>`_.

Developer notes
===============
The application may be profiled with annotations by specifying USE_NTVX to cmake and providing the path to the stand-alone nvtx header via NVTX_HEADER_DIR. Regions covered by the 'Monitor' class in cuda code will automatically appear in the nsight profiler.

**********
References
**********
`Mitchell R, Frank E. (2017) Accelerating the XGBoost algorithm using GPU computing. PeerJ Computer Science 3:e127 https://doi.org/10.7717/peerj-cs.127 <https://peerj.com/articles/cs-127/>`_

`Nvidia Parallel Forall: Gradient Boosting, Decision Trees and XGBoost with CUDA <https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/>`_

`Out-of-Core GPU Gradient Boosting <https://arxiv.org/abs/2005.09148>`_

Contributors
============
Many thanks to the following contributors (alphabetical order):

* Andrey Adinets
* Jiaming Yuan
* Jonathan C. McKinney
* Matthew Jones
* Philip Cho
* Rong Ou
* Rory Mitchell
* Shankara Rao Thejaswi Nanditale
* Sriram Chandramouli
* Vinay Deshpande

Please report bugs to the XGBoost issues list: https://github.com/dmlc/xgboost/issues.  For general questions please visit our user form: https://discuss.xgboost.ai/.
