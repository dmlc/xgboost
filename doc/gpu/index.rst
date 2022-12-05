###################
XGBoost GPU Support
###################

This page contains information about GPU algorithms supported in XGBoost.

.. note:: CUDA 11.0, Compute Capability 5.0 required (See `this list <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ to look up compute capability of your GPU card.)

*********************************************
CUDA Accelerated Tree Construction Algorithms
*********************************************

Most of the algorithms in XGBoost including training, prediction and evaluation can be accelerated with CUDA-capable GPUs.

Usage
=====
Specify the ``tree_method`` parameter as ``gpu_hist``. For details around the ``tree_method`` parameter, see :doc:`tree method </treemethod>`.

Supported parameters
--------------------

GPU accelerated prediction is enabled by default for the above mentioned ``tree_method`` parameters but can be switched to CPU prediction by setting ``predictor`` to ``cpu_predictor``. This could be useful if you want to conserve GPU memory. Likewise when using CPU algorithms, GPU accelerated prediction can be enabled by setting ``predictor`` to ``gpu_predictor``.

The device ordinal (which GPU to use if you have many of them) can be selected using the
``gpu_id`` parameter, which defaults to 0 (the first device reported by CUDA runtime).


The GPU algorithms currently work with CLI, Python, R, and JVM packages. See :doc:`/install` for details.

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
<https://github.com/dmlc/xgboost/tree/master/demo/gpu_acceleration>`__.

Multi-node Multi-GPU Training
=============================

XGBoost supports fully distributed GPU training using `Dask <https://dask.org/>`_, ``Spark`` and ``PySpark``. For getting started with Dask see our tutorial :doc:`/tutorials/dask` and worked examples `here <https://github.com/dmlc/xgboost/tree/master/demo/dask>`__, also Python documentation :ref:`dask_api` for complete reference. For usage with ``Spark`` using Scala see :doc:`/jvm/xgboost4j_spark_gpu_tutorial`. Lastly for distributed GPU training with ``PySpark``, see :doc:`/tutorials/spark_estimator`.


Memory usage
============
The following are some guidelines on the device memory usage of the `gpu_hist` tree method.

Memory inside xgboost training is generally allocated for two reasons - storing the dataset and working memory.

The dataset itself is stored on device in a compressed ELLPACK format. The ELLPACK format is a type of sparse matrix that stores elements with a constant row stride. This format is convenient for parallel computation when compared to CSR because the row index of each element is known directly from its address in memory. The disadvantage of the ELLPACK format is that it becomes less memory efficient if the maximum row length is significantly more than the average row length. Elements are quantised and stored as integers. These integers are compressed to a minimum bit length. Depending on the number of features, we usually don't need the full range of a 32 bit integer to store elements and so compress this down. The compressed, quantised ELLPACK format will commonly use 1/4 the space of a CSR matrix stored in floating point.

Working memory is allocated inside the algorithm proportional to the number of rows to keep track of gradients, tree positions and other per row statistics. Memory is allocated for histogram bins proportional to the number of bins, number of features and nodes in the tree. For performance reasons we keep histograms in memory from previous nodes in the tree, when a certain threshold of memory usage is passed we stop doing this to conserve memory at some performance loss.

If you are getting out-of-memory errors on a big dataset, try the or :py:class:`xgboost.QuantileDMatrix` or :doc:`external memory version </tutorials/external_memory>`. Note that when ``external memory`` is used for GPU hist, it's best to employ gradient based sampling as well. Last but not least, ``inplace_predict`` can be preferred over ``predict`` when data is already on GPU. Both ``QuantileDMatrix`` and ``inplace_predict`` are automatically enabled if you are using the scikit-learn interface.


CPU-GPU Interoperability
========================
XGBoost models trained on GPUs can be used on CPU-only systems to generate predictions. For information about how to save and load an XGBoost model, see :doc:`/tutorials/saving_model`.


Developer notes
===============
The application may be profiled with annotations by specifying USE_NTVX to cmake. Regions covered by the 'Monitor' class in CUDA code will automatically appear in the nsight profiler when `verbosity` is set to 3.

**********
References
**********
`Mitchell R, Frank E. (2017) Accelerating the XGBoost algorithm using GPU computing. PeerJ Computer Science 3:e127 https://doi.org/10.7717/peerj-cs.127 <https://peerj.com/articles/cs-127/>`_

`NVIDIA Parallel Forall: Gradient Boosting, Decision Trees and XGBoost with CUDA <https://devblogs.nvidia.com/parallelforall/gradient-boosting-decision-trees-xgboost-cuda/>`_

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
