#####################################
Using XGBoost External Memory Version
#####################################

When working with large datasets, training XGBoost models can be challenging as the entire
dataset needs to be loaded into memory. This can be costly and sometimes
infeasible. Starting from 1.5, users can define a custom iterator to load data in chunks
for running XGBoost algorithms. External memory can be used for training and prediction,
but training is the primary use case and it will be our focus in this tutorial. For
prediction and evaluation, users can iterate through the data themselves, whereas training
requires the entire dataset to be loaded into the memory. Significant progress was made in
the 3.0 release for the GPU implementation. We will introduce the difference between CPU
and GPU in the following sections.

.. note::

   Training on data from external memory is not supported by the ``exact`` tree method.

.. note::

   The feature is considered experimental but ready for public testing in 3.0. Vector-leaf
   is not yet supported.

The external memory support has undergone multiple development iterations. Like the
:py:class:`~xgboost.QuantileDMatrix` with :py:class:`~xgboost.DataIter`, XGBoost loads
data batch-by-batch using a custom iterator supplied by the user. However, unlike the
:py:class:`~xgboost.QuantileDMatrix`, external memory does not concatenate the batches
(unless specified by the ``extmem_single_page``) . Instead, it caches all batches in the
external memory and fetch them on-demand. Go to the end of the document to see a
comparison between :py:class:`~xgboost.QuantileDMatrix` and the external memory version of
:py:class:`~xgboost.ExtMemQuantileDMatrix`.

**Contents**

.. contents::
  :backlinks: none
  :local:

*************
Data Iterator
*************

Starting with XGBoost 1.5, users can define their own data loader using Python or C
interface. Some examples are in the ``demo`` directory for a quick start. To enable
external memory training, users need to define a data iterator with 2 class methods:
``next`` and ``reset``, then pass it into the :py:class:`~xgboost.DMatrix` or the
:py:class:`~xgboost.ExtMemQuantileDMatrix` constructor.

.. code-block:: python

  import os
  from typing import List, Callable
  import xgboost
  from sklearn.datasets import load_svmlight_file

  class Iterator(xgboost.DataIter):
    def __init__(self, svm_file_paths: List[str]) -> None:
      self._file_paths = svm_file_paths
      self._it = 0
      # XGBoost will generate some cache files under the current directory with the prefix
      # "cache"
      super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data: Callable) -> bool:
      """Advance the iterator by 1 step and pass the data to XGBoost. This function is
      called by XGBoost during the construction of ``DMatrix``

      """
      if self._it == len(self._file_paths):
        # return False to let XGBoost know this is the end of the iteration
        return False

      # input_data is a function passed in by XGBoost and has the exact same signature of
      # ``DMatrix``
      X, y = load_svmlight_file(self._file_paths[self._it])
      # Keyword-only arguments, see the ``DMatrix`` class for accepted arguments.
      input_data(data=X, label=y)
      self._it += 1
      # Return True to let XGBoost know we haven't seen all the files yet.
      return True

    def reset(self) -> None:
      """Reset the iterator to its beginning"""
      self._it = 0

  it = Iterator(["file_0.svm", "file_1.svm", "file_2.svm"])

  # Use the ``ExtMemQuantileDMatrix`` for the hist tree method.
  Xy = xgboost.ExtMemQuantileDMatrix(it)
  booster = xgboost.train({"tree_method": "hist"}, Xy)

  # The ``approx`` tree method also works, but with lower performance and cannot be used
  # with the quantile DMatrix.
  Xy = xgboost.DMatrix(it)
  booster = xgboost.train({"tree_method": "approx"}, Xy)

The above snippet is a simplified version of :ref:`sphx_glr_python_examples_external_memory.py`.
For an example in C, please see ``demo/c-api/external-memory/``. The iterator is the
common interface for using external memory with XGBoost, you can pass the resulting
:py:class:`~xgboost.DMatrix` object for training, prediction, and evaluation.

The :py:class:`~xgboost.ExtMemQuantileDMatrix` is an external memory version of the
:py:class:`~xgboost.QuantileDMatrix`. These two classes are specifically designed for the
``hist`` tree method for reduced memory usage and data loading overhead. See respective
references for more info.

It is important to set the batch size based on the memory available. A good starting point
for CPU is to set the batch size to 10GB per batch if you have 64GB of memory. It is *not*
recommended to set small batch sizes like 32 samples per batch, as this can severely hurt
performance in gradient boosting. See below sections for information about the GPU version
and other best practices.

**********************************
GPU Version (GPU Hist tree method)
**********************************

External memory is supported by GPU algorithms (i.e., when ``device`` is set to
``cuda``). Starting with 3.0, the default GPU implementation is similar to what the CPU
version does. It also supports the use of :py:class:`~xgboost.ExtMemQuantileDMatrix` when
the ``hist`` tree method is employed. For a GPU device, the main memory is the device
memory, whereas the external memory can be either a disk or the CPU memory. XGBoost stages
the cache on CPU memory by default. Users can change the backing storage to disk by
specifying the ``on_host`` parameter in the :py:class:`~xgboost.DataIter`. However, using
the disk is not recommended as it's likely to make the GPU slower than the CPU. The option
is here for experimental purposes only. In addition,
:py:class:`~xgboost.ExtMemQuantileDMatrix` parameters ``max_num_device_pages``,
``min_cache_page_bytes``, and ``max_quantile_batches`` can help control the data placement
and memory usage.

Inputs to the :py:class:`~xgboost.ExtMemQuantileDMatrix` (through the iterator) must be on
the GPU. Following is a snippet from :ref:`sphx_glr_python_examples_external_memory.py`:

.. code-block:: python

    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    # It's important to use RMM for GPU-based external memory to improve performance.
    # If XGBoost is not built with RMM support, a warning will be raised.
    # We use the pool memory resource here, you can also try the `ArenaMemoryResource` for
    # improved memory fragmentation handling.
    mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource())
    rmm.mr.set_current_device_resource(mr)
    # Set the allocator for cupy as well.
    cp.cuda.set_allocator(rmm_cupy_allocator)
    # Make sure XGBoost is using RMM for all allocations.
    with xgboost.config_context(use_rmm=True):
        # Construct the iterators for ExtMemQuantileDMatrix
	# ...
	# Build the ExtMemQuantileDMatrix and start training
	Xy_train = xgboost.ExtMemQuantileDMatrix(it_train, max_bin=n_bins)
	# Use the training DMatrix as a reference
	Xy_valid = xgboost.ExtMemQuantileDMatrix(it_valid, max_bin=n_bins, ref=Xy_train)
	booster = xgboost.train(
	    {
		"tree_method": "hist",
		"max_depth": 6,
		"max_bin": n_bins,
		"device": device,
	    },
	    Xy_train,
	    num_boost_round=n_rounds,
	    evals=[(Xy_train, "Train"), (Xy_valid, "Valid")]
	)

It's crucial to use `RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`__ with
an asynchronous memory resource for all memory allocation when training with external
memory. XGBoost relies on the asynchronous memory pool to reduce the overhead of data
fetching. In addition, the open source `NVIDIA Linux driver
<https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/>`__
is required for ``Heterogeneous memory management (HMM)`` support. Usually, users need not
to change :py:class:`~xgboost.ExtMemQuantileDMatrix` parameters ``max_num_device_pages``
and ``min_cache_page_bytes``, they are automatically configured based on the device and
don't change model accuracy. However, the ``max_quantile_batches`` can be useful if
:py:class:`~xgboost.ExtMemQuantileDMatrix` is running out of device memory during
construction, see :py:class:`~xgboost.QuantileDMatrix` and the following sections for more
info.

In addition to the batch-based data fetching, the GPU version supports concatenating
batches into a single blob for the training data to improve performance. For GPUs
connected via PCIe instead of nvlink, the performance overhead with batch-based training
is significant, particularly for non-dense data. Overall, it can be at least five times
slower than in-core training. Concatenating pages can be used to get the performance
closer to in-core training. This option should be used in combination with subsampling to
reduce the memory usage. During concatenation, subsampling removes a portion of samples,
reducing the training dataset size. The GPU hist tree method supports `gradient-based
sampling`, enabling users to set a low sampling rate without compromising accuracy. Before
3.0, concatenation with subsampling was the only option for GPU-based external
memory. After 3.0, XGBoost uses the regular batch fetching as the default while the page
concatenation can be enabled by:

.. code-block:: python

  param = {
    "device": "cuda",
    "extmem_single_page": true,
    'subsample': 0.2,
    'sampling_method': 'gradient_based',
  }

For more information about the sampling algorithm and its use in external memory training,
see `this paper <https://arxiv.org/abs/2005.09148>`_. Lastly, see following sections for
best practices.

==========
NVLink-C2C
==========

The newer NVIDIA platforms like `Grace-Hopper
<https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/>`__ use `NVLink-C2C
<https://www.nvidia.com/en-us/data-center/nvlink-c2c/>`__, which facilitates a fast
interconnect between the CPU and the GPU. With the host memory serving as the data cache,
XGBoost can retrieve data with significantly lower overhead. When the input data is dense,
there's minimal to no performance loss for training, except for the initial construction
of the :py:class:`~xgboost.ExtMemQuantileDMatrix`.  The initial construction iterates
through the input data twice, as a result, the most significant overhead compared to
in-core training is one additional data read when the data is dense. Please note that
there are multiple variants of the platform and they come with different C2C
bandwidths. During initial development of the feature, we used the LPDDR5 480G version,
which has about 350GB/s bandwidth for host to device transfer.

To run experiments on these platforms, the open source `NVIDIA Linux driver
<https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/>`__
with version ``>=565.47`` is required, it should come with CTK 12.7 and later versions.

********************
Distributed Training
********************

Distributed training is similar to in-core learning, but the work for framework
integration is still on-going. See :ref:`sphx_glr_python_examples_distributed_extmem_basic.py`
for an example for using the communicator to build a simple pipeline. Since users can
define their custom data loader, it's unlikely that existing distributed frameworks
interface in XGBoost can meet all the use cases, the example can be a starting point for
users who have custom infrastructure.

**************
Best Practices
**************

In previous sections, we demonstrated how to train a tree-based model with data residing
on an external memory and made some recommendations for batch size. Here are some other
configurations we find useful. The external memory feature involves iterating through data
batches stored in a cache during tree construction. For optimal performance, we recommend
using the ``grow_policy=depthwise`` setting, which allows XGBoost to build an entire layer
of tree nodes with only a few batch iterations. Conversely, using the ``lossguide`` policy
requires XGBoost to iterate over the data set for each tree node, resulting in
significantly slower performance.

In addition, this ``hist`` tree method should be preferred over the ``approx`` tree method
as the former doesn't recreate the histogram bins for every iteration. Creating the
histogram bins requires loading the raw input data, which is prohibitively expensive. The
:py:class:`~xgboost.ExtMemQuantileDMatrix` designed for the ``hist`` tree method can speed
up the initial data construction and the evaluation significantly for external memory.

Since the external memory implementation focuses on training where XGBoost needs to access
the entire dataset, only the ``X`` is divided into batches while everything else is
concatenated. As a result, it's recommended for users to define their own management code
to iterate through the data for inference, especially for SHAP value computation. The size
of SHAP results can be larger than ``X``, making external memory in XGBoost less
effective. Some frameworks like ``dask`` can help with the data chunking and iterate
through the data for inference with memory spilling.

When external memory is used, the performance of CPU training is limited by disk IO
(input/output) speed. This means that the disk IO speed primarily determines the training
speed. Similarly, PCIe bandwidth limits the GPU performance, assuming the CPU memory is
used as a cache and address translation services (ATS) is unavailable. During development,
we observed that typical data transfer in XGBoost with PCIe4x16 has about 24GB/s
bandwidth, which is significantly lower than the GPU processing performance. Whereas with
a C2C-enabled machine, the performance of data transfer and processing in training are
similar. Running inference is much less computation-intensive than training and, hence,
much faster. As a result, the performance bottleneck of inference is back to data
transfer. For GPU, the time it takes to read the data from host to device completely
determines the time it takes to run inference, even if a C2C link is available.

.. code-block:: python

    Xy_train = xgboost.ExtMemQuantileDMatrix(it_train, max_bin=n_bins)
    Xy_valid = xgboost.ExtMemQuantileDMatrix(it_valid, max_bin=n_bins, ref=Xy_train)

In addition, since the GPU implementation relies on asynchronous memory pool, which is
subject to memory fragmentation even if the :py:class:`~rmm.mr.CudaAsyncMemoryResource` is
used. You might want to start the training with a fresh pool instead of starting training
right after the ETL process. If you run into out-of-memory errors and you are convinced
that the pool is not full yet (pool memory usage can be profiled with ``nsight-system``),
consider tuning the RMM memory resource like using
:py:class:`~rmm.mr.CudaAsyncMemoryResource` in conjunction with
:py:class:`BinningMemoryResource(mr, 21, 25) <rmm.mr.BinningMemoryResource>` instead of
the :py:class:`~rmm.mr.PoolMemoryResource`. Alternately, the
:py:class:`~rmm.mr.ArenaMemoryResource` is also an excellent option.

During CPU benchmarking, we used an NVMe connected to a PCIe-4 slot. Other types of
storage can be too slow for practical usage. However, your system will likely perform some
caching to reduce the overhead of the file read. See the following sections for remarks.

.. _ext_remarks:

*******
Remarks
*******

When using external memory with XGBoost, data is divided into smaller chunks so that only
a fraction of it needs to be stored in memory at any given time. It's important to note
that this method only applies to the predictor data (``X``), while other data, like labels
and internal runtime structures are concatenated. This means that memory reduction is most
effective when dealing with wide datasets where ``X`` is significantly larger in size
compared to other data like ``y``, while it has little impact on slim datasets.

As one might expect, fetching data on demand puts significant pressure on the storage
device. Today's computing devices can process way more data than storage devices can read
in a single unit of time. The ratio is in the order of magnitudes. A GPU is capable of
processing hundreds of Gigabytes of floating-point data in a split second. On the other
hand, a four-lane NVMe storage connected to a PCIe-4 slot usually has about 6GB/s of data
transfer rate. As a result, the training is likely to be severely bounded by your storage
device. Before adopting the external memory solution, some back-of-envelop calculations
might help you determine its viability. For instance, if your NVMe drive can transfer 4GB
(a reasonably practical number) of data per second, and you have a 100GB of data in a
compressed XGBoost cache (corresponding to a dense float32 numpy array with 200GB, give or
take). A tree with depth 8 needs at least 16 iterations through the data when the
parameter is optimal. You need about 14 minutes to train a single tree without accounting
for some other overheads and assume the computation overlaps with the IO. If your dataset
happens to have a TB-level size, you might need thousands of trees to get a generalized
model. These calculations can help you get an estimate of the expected training time.

However, sometimes, we can ameliorate this limitation. One should also consider that the
OS (mainly talking about the Linux kernel) can usually cache the data on host memory. It
only evicts pages when new data comes in and there's no room left. In practice, at least
some portion of the data can persist in the host memory throughout the entire training
session. We are aware of this cache when optimizing the external memory fetcher. The
compressed cache is usually smaller than the raw input data, especially when the input is
dense without any missing value. If the host memory can fit a significant portion of this
compressed cache, the performance should be decent after initialization. Our development
so far focuses on following fronts of optimization for external memory:

- Avoid iterating through the data whenever appropriate.
- If the OS can cache the data, the performance should be close to in-core training.
- For GPU, the actual computation should overlap with memory copy as much as possible.

Starting with XGBoost 2.0, the implementation of external memory uses ``mmap``. It has not
been tested against system errors like disconnected network devices (`SIGBUS`). In the
face of a bus error, you will see a hard crash and need to clean up the cache files. If
the training session might take a long time and you use solutions like NVMe-oF, we
recommend checkpointing your model periodically. Also, it's worth noting that most tests
have been conducted on Linux distributions.

Another important point to keep in mind is that creating the initial cache for XGBoost may
take some time. The interface to external memory is through custom iterators, which we can
not assume to be thread-safe. Therefore, initialization is performed sequentially. Using
the :py:func:`~xgboost.config_context` with `verbosity=2` can give you some information on
what XGBoost is doing during the wait if you don't mind the extra output.

*******************************
Compared to the QuantileDMatrix
*******************************

Passing an iterator to the :py:class:`~xgboost.QuantileDMatrix` enables direct
construction of :py:class:`~xgboost.QuantileDMatrix` with data chunks. On the other hand,
if it's passed to the :py:class:`~xgboost.DMatrix` or the
:py:class:`~xgboost.ExtMemQuantileDMatrix`, it instead enables the external memory
feature. The :py:class:`~xgboost.QuantileDMatrix` concatenates the data in memory after
compression and doesn't fetch data during training. On the other hand, the external memory
:py:class:`~xgboost.DMatrix` (:py:class:`~xgboost.ExtMemQuantileDMatrix`) fetches data
batches from external memory on demand. Use the :py:class:`~xgboost.QuantileDMatrix` (with
iterator if necessary) when you can fit most of your data in memory. For many platforms,
the training speed can be an order of magnitude faster than external memory.

*************
Brief History
*************

For a long time, external memory support has been an experimental feature and has
undergone multiple development iterations. Here's a brief summary of major changes:

- Gradient-based sampling was introduced to the GPU hist in 1.1.
- The iterator interface was introduced in 1.5, along with a major rewrite for the
  internal framework.
- 2.0 introduced the use of ``mmap``, along with optimization in XBGoost to enable
  zero-copy data fetching.
- 3.0 reworked the GPU implementation to support caching data on the host and disk,
  introduced the :py:class:`~xgboost.ExtMemQuantileDMatrix` class, added quantile-based
  objectives support.
- In addition, we begin support for distributed training in 3.0

****************
Text File Inputs
****************

.. warning::

   This is the original form of external memory support before 1.5 and is now deprecated,
   users are encouraged to use a custom data iterator instead.

There is no significant difference between using the external memory version of text input
and the in-memory version of text input. The only difference is the filename format.

The external memory version takes in the following `URI
<https://en.wikipedia.org/wiki/Uniform_Resource_Identifier>`_ format:

.. code-block:: none

  filename?format=libsvm#cacheprefix

The ``filename`` is the typical path to LIBSVM format file you want to load in, and
``cacheprefix`` is a path to a cache file that XGBoost will use for caching preprocessed
data in binary form.

To load from csv files, use the following syntax:

.. code-block:: none

  filename.csv?format=csv&label_column=0#cacheprefix

where ``label_column`` should point to the csv column acting as the label.

If you have a dataset stored in a file similar to ``demo/data/agaricus.txt.train`` with LIBSVM
format, the external memory support can be enabled by:

.. code-block:: python

  dtrain = DMatrix('../data/agaricus.txt.train?format=libsvm#dtrain.cache')

XGBoost will first load ``agaricus.txt.train`` in, preprocess it, then write to a new file named
``dtrain.cache`` as an on disk cache for storing preprocessed data in an internal binary format. For
more notes about text input formats, see :doc:`/tutorials/input_format`.

For the CLI version, simply add the cache suffix, e.g. ``"../data/agaricus.txt.train?format=libsvm#dtrain.cache"``.
