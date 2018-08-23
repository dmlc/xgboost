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

+--------------------------+---------------+--------------+
| parameter                | ``gpu_exact`` | ``gpu_hist`` |
+==========================+===============+==============+
| ``subsample``            | |cross|       | |tick|       |
+--------------------------+---------------+--------------+
| ``colsample_bytree``     | |cross|       | |tick|       |
+--------------------------+---------------+--------------+
| ``colsample_bylevel``    | |cross|       | |tick|       |
+--------------------------+---------------+--------------+
| ``max_bin``              | |cross|       | |tick|       |
+--------------------------+---------------+--------------+
| ``gpu_id``               | |tick|        | |tick|       |
+--------------------------+---------------+--------------+
| ``n_gpus``               | |cross|       | |tick|       |
+--------------------------+---------------+--------------+
| ``predictor``            | |tick|        | |tick|       |
+--------------------------+---------------+--------------+
| ``grow_policy``          | |cross|       | |tick|       |
+--------------------------+---------------+--------------+
| ``monotone_constraints`` | |cross|       | |tick|       |
+--------------------------+---------------+--------------+

GPU accelerated prediction is enabled by default for the above mentioned ``tree_method`` parameters but can be switched to CPU prediction by setting ``predictor`` to ``cpu_predictor``. This could be useful if you want to conserve GPU memory. Likewise when using CPU algorithms, GPU accelerated prediction can be enabled by setting ``predictor`` to ``gpu_predictor``.

The device ordinal can be selected using the ``gpu_id`` parameter, which defaults to 0.

Multiple GPUs can be used with the ``gpu_hist`` tree method using the ``n_gpus`` parameter. which defaults to 1. If this is set to -1 all available GPUs will be used.  If ``gpu_id`` is specified as non-zero, the gpu device order is ``mod(gpu_id + i) % n_visible_devices`` for ``i=0`` to ``n_gpus-1``.  As with GPU vs. CPU, multi-GPU will not always be faster than a single GPU due to PCI bus bandwidth that can limit performance.

.. note:: Enabling multi-GPU training

  Default installation may not enable multi-GPU training. To use multiple GPUs, make sure to read :ref:`build_gpu_support`.

The GPU algorithms currently work with CLI, Python and R packages. See :doc:`/build` for details.

.. code-block:: python
  :caption: Python example

  param['gpu_id'] = 0
  param['max_bin'] = 16
  param['tree_method'] = 'gpu_hist'

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

**********
References
**********
`Mitchell R, Frank E. (2017) Accelerating the XGBoost algorithm using GPU computing. PeerJ Computer Science 3:e127 https://doi.org/10.7717/peerj-cs.127 <https://peerj.com/articles/cs-127/>`_

`Nvidia Parallel Forall: Gradient Boosting, Decision Trees and XGBoost with CUDA <https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/>`_

Authors
=======
* Rory Mitchell
* Jonathan C. McKinney
* Shankara Rao Thejaswi Nanditale
* Vinay Deshpande
* ... and the rest of the H2O.ai and NVIDIA team.

Please report bugs to the user forum https://discuss.xgboost.ai/.

