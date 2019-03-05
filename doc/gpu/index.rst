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

+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| tree_method  | Description                                                                                                                                                           |
+==============+=======================================================================================================================================================================+
| gpu_exact    | The standard XGBoost tree construction algorithm. Performs exact search for splits. Slower and uses considerably more memory than ``gpu_hist``.                       |
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| gpu_hist     | Equivalent to the XGBoost fast histogram algorithm. Much faster and uses considerably less memory. NOTE: Will run very slowly on GPUs older than Pascal architecture. |
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Supported parameters
--------------------

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+--------------------------------+---------------+--------------+
| parameter                      | ``gpu_exact`` | ``gpu_hist`` |
+================================+===============+==============+
| ``subsample``                  | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+
| ``colsample_bytree``           | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+
| ``colsample_bylevel``          | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+
| ``max_bin``                    | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+
| ``gpu_id``                     | |tick|        | |tick|       |
+--------------------------------+---------------+--------------+
| ``n_gpus``                     | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+
| ``predictor``                  | |tick|        | |tick|       |
+--------------------------------+---------------+--------------+
| ``grow_policy``                | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+
| ``monotone_constraints``       | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+
| ``single_precision_histogram`` | |cross|       | |tick|       |
+--------------------------------+---------------+--------------+

GPU accelerated prediction is enabled by default for the above mentioned ``tree_method`` parameters but can be switched to CPU prediction by setting ``predictor`` to ``cpu_predictor``. This could be useful if you want to conserve GPU memory. Likewise when using CPU algorithms, GPU accelerated prediction can be enabled by setting ``predictor`` to ``gpu_predictor``.

The experimental parameter ``single_precision_histogram`` can be set to True to enable building histograms using single precision. This may improve speed, in particular on older architectures.

The device ordinal can be selected using the ``gpu_id`` parameter, which defaults to 0.

Multiple GPUs can be used with the ``gpu_hist`` tree method using the ``n_gpus`` parameter. which defaults to 1. If this is set to -1 all available GPUs will be used.  If ``gpu_id`` is specified as non-zero, the selected gpu devices will be from ``gpu_id`` to ``gpu_id+n_gpus``, please note that ``gpu_id+n_gpus`` must be less than or equal to the number of available GPUs on your system.  As with GPU vs. CPU, multi-GPU will not always be faster than a single GPU due to PCI bus bandwidth that can limit performance.

.. note:: Enabling multi-GPU training

  Default installation may not enable multi-GPU training. To use multiple GPUs, make sure to read :ref:`build_gpu_support`.

The GPU algorithms currently work with CLI, Python and R packages. See :doc:`/build` for details.

.. code-block:: python
  :caption: Python example

  param['gpu_id'] = 0
  param['max_bin'] = 16
  param['tree_method'] = 'gpu_hist'

Objective functions
===================
Most of the objective functions implemented in XGBoost can be run on GPU.  Following table shows current support status.

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+-----------------+-------------+
| Objectives      | GPU support |
+-----------------+-------------+
| reg:linear      | |tick|      |
+-----------------+-------------+
| reg:logistic    | |tick|      |
+-----------------+-------------+
| binary:logistic | |tick|      |
+-----------------+-------------+
| binary:logitraw | |tick|      |
+-----------------+-------------+
| binary:hinge    | |tick|      |
+-----------------+-------------+
| count:poisson   | |tick|      |
+-----------------+-------------+
| reg:gamma       | |tick|      |
+-----------------+-------------+
| reg:tweedie     | |tick|      |
+-----------------+-------------+
| multi:softmax   | |tick|      |
+-----------------+-------------+
| multi:softprob  | |tick|      |
+-----------------+-------------+
| survival:cox    | |cross|     |
+-----------------+-------------+
| rank:pairwise   | |cross|     |
+-----------------+-------------+
| rank:ndcg       | |cross|     |
+-----------------+-------------+
| rank:map        | |cross|     |
+-----------------+-------------+

For multi-gpu support, objective functions also honor the ``n_gpus`` parameter,
which, by default is set to 1.  To disable running objectives on GPU, just set
``n_gpus`` to 0.

Metric functions
===================
Following table shows current support status for evaluation metrics on the GPU.

.. |tick| unicode:: U+2714
.. |cross| unicode:: U+2718

+-----------------+-------------+
| Metric          | GPU Support |
+=================+=============+
| rmse            | |tick|      |
+-----------------+-------------+
| mae             | |tick|      |
+-----------------+-------------+
| logloss         | |tick|      |
+-----------------+-------------+
| error           | |tick|      |
+-----------------+-------------+
| merror          | |cross|     |
+-----------------+-------------+
| mlogloss        | |cross|     |
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

As for objective functions, metrics honor the ``n_gpus`` parameter,
which, by default is set to 1.  To disable running metrics on GPU, just set
``n_gpus`` to 0.


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
| gpu_exact    | 161.08   |
+--------------+----------+
| exact        | 1082.20  |
+--------------+----------+

See `GPU Accelerated XGBoost <https://xgboost.ai/2016/12/14/GPU-accelerated-xgboost.html>`_ and `Updates to the XGBoost GPU algorithms <https://xgboost.ai/2018/07/04/gpu-xgboost-update.html>`_ for additional performance benchmarks of the ``gpu_exact`` and ``gpu_hist`` tree methods.

Developer notes
==========
The application may be profiled with annotations by specifying USE_NTVX to cmake and providing the path to the stand-alone nvtx header via NVTX_HEADER_DIR. Regions covered by the 'Monitor' class in cuda code will automatically appear in the nsight profiler.

**********
References
**********
`Mitchell R, Frank E. (2017) Accelerating the XGBoost algorithm using GPU computing. PeerJ Computer Science 3:e127 https://doi.org/10.7717/peerj-cs.127 <https://peerj.com/articles/cs-127/>`_

`Nvidia Parallel Forall: Gradient Boosting, Decision Trees and XGBoost with CUDA <https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/>`_

Contributors
=======
Many thanks to the following contributors (alphabetical order):
* Andrey Adinets
* Jiaming Yuan
* Jonathan C. McKinney
* Matthew Jones
* Philip Cho
* Rory Mitchell
* Shankara Rao Thejaswi Nanditale
* Vinay Deshpande

Please report bugs to the user forum https://discuss.xgboost.ai/.
