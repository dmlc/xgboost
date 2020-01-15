###################
XGBoost GPU Support
###################

This page contains information about GPU algorithms supported in XGBoost.
To install GPU support, checkout the :doc:`/build`.

.. note:: CUDA 8.0, Compute Capability 3.5 required

  The GPU algorithms in XGBoost require a graphics card with compute capability 3.5 or higher, with
  CUDA toolkits 8.0 or later.
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
| gpu_hist              | Equivalent to the XGBoost fast histogram algorithm. Much faster and uses considerably less memory. NOTE: Will run very slowly on GPUs older than Pascal architecture. |
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
| ``n_gpus`` (deprecated)        | |tick|       |
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


Single Node Multi-GPU
=====================
.. note:: Single node multi-GPU training with `n_gpus` parameter is deprecated after 0.90.  Please use distributed GPU training with one process per GPU.

Multi-node Multi-GPU Training
=============================
XGBoost supports fully distributed GPU training using `Dask <https://dask.org/>`_. For
getting started see our tutorial :doc:`/tutorials/dask` and worked examples `here
<https://github.com/dmlc/xgboost/tree/master/demo/dask>`_, also Python documentation
:ref:`dask_api` for complete reference.


Objective functions
===================
Most of the objective functions implemented in XGBoost can be run on GPU.  Following table shows current support status.

+--------------------+-------------+
| Objectives         | GPU support |
+--------------------+-------------+
| reg:squarederror   | |tick|      |
+--------------------+-------------+
| reg:squaredlogerror| |tick|      |
+--------------------+-------------+
| reg:logistic       | |tick|      |
+--------------------+-------------+
| binary:logistic    | |tick|      |
+--------------------+-------------+
| binary:logitraw    | |tick|      |
+--------------------+-------------+
| binary:hinge       | |tick|      |
+--------------------+-------------+
| count:poisson      | |tick|      |
+--------------------+-------------+
| reg:gamma          | |tick|      |
+--------------------+-------------+
| reg:tweedie        | |tick|      |
+--------------------+-------------+
| multi:softmax      | |tick|      |
+--------------------+-------------+
| multi:softprob     | |tick|      |
+--------------------+-------------+
| survival:cox       | |cross|     |
+--------------------+-------------+
| rank:pairwise      | |cross|     |
+--------------------+-------------+
| rank:ndcg          | |cross|     |
+--------------------+-------------+
| rank:map           | |cross|     |
+--------------------+-------------+

Objective will run on GPU if GPU updater (``gpu_hist``), otherwise they will run on CPU by
default.  For unsupported objectives XGBoost will fall back to using CPU implementation by
default.

Metric functions
===================
Following table shows current support status for evaluation metrics on the GPU.

+-----------------+-------------+
| Metric          | GPU Support |
+=================+=============+
| rmse            | |tick|      |
+-----------------+-------------+
| rmsle           | |tick|      |
+-----------------+-------------+
| mae             | |tick|      |
+-----------------+-------------+
| logloss         | |tick|      |
+-----------------+-------------+
| error           | |tick|      |
+-----------------+-------------+
| merror          | |tick|      |
+-----------------+-------------+
| mlogloss        | |tick|      |
+-----------------+-------------+
| auc             | |cross|     |
+-----------------+-------------+
| aucpr           | |cross|     |
+-----------------+-------------+
| ndcg            | |cross|     |
+-----------------+-------------+
| map             | |cross|     |
+-----------------+-------------+
| poisson-nloglik | |tick|      |
+-----------------+-------------+
| gamma-nloglik   | |tick|      |
+-----------------+-------------+
| cox-nloglik     | |cross|     |
+-----------------+-------------+
| gamma-deviance  | |tick|      |
+-----------------+-------------+
| tweedie-nloglik | |tick|      |
+-----------------+-------------+

Similar to objective functions, default device for metrics is selected based on tree
updater and predictor (which is selected based on tree updater).

Benchmarks
==========
You can run benchmarks on synthetic data for binary classification:

.. code-block:: bash

  python tests/benchmark/benchmark.py

Training time time on 1,000,000 rows x 50 columns with 500 boosting iterations and 0.25/0.75 test/train split on i7-6700K CPU @ 4.00GHz and Pascal Titan X yields the following results:

+--------------+----------+
| tree_method  | Time (s) |
+==============+==========+
| gpu_hist     | 13.87    |
+--------------+----------+
| hist         | 63.55    |
+--------------+----------+
| exact        | 1082.20  |
+--------------+----------+

See `GPU Accelerated XGBoost <https://xgboost.ai/2016/12/14/GPU-accelerated-xgboost.html>`_ and `Updates to the XGBoost GPU algorithms <https://xgboost.ai/2018/07/04/gpu-xgboost-update.html>`_ for additional performance benchmarks of the ``gpu_hist`` tree method.

Developer notes
===============
The application may be profiled with annotations by specifying USE_NTVX to cmake and providing the path to the stand-alone nvtx header via NVTX_HEADER_DIR. Regions covered by the 'Monitor' class in cuda code will automatically appear in the nsight profiler.

**********
References
**********
`Mitchell R, Frank E. (2017) Accelerating the XGBoost algorithm using GPU computing. PeerJ Computer Science 3:e127 https://doi.org/10.7717/peerj-cs.127 <https://peerj.com/articles/cs-127/>`_

`Nvidia Parallel Forall: Gradient Boosting, Decision Trees and XGBoost with CUDA <https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/>`_

Contributors
============
Many thanks to the following contributors (alphabetical order):

* Andrey Adinets
* Jiaming Yuan
* Jonathan C. McKinney
* Matthew Jones
* Philip Cho
* Rory Mitchell
* Shankara Rao Thejaswi Nanditale
* Vinay Deshpande

Please report bugs to the XGBoost issues list: https://github.com/dmlc/xgboost/issues.  For general questions please visit our user form: https://discuss.xgboost.ai/.
