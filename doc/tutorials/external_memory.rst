#####################################
Using XGBoost External Memory Version
#####################################

When working with large datasets, training XGBoost models can be challenging as the entire
dataset needs to be loaded into memory. This can be costly and sometimes
infeasible. Staring from 1.5, users can define a custom iterator to load data in chunks
for running XGBoost algorithms. External memory can be used for both training and
prediction, but training is the primary use case and it will be our focus in this
tutorial. For prediction and evaluation, users can iterate through the data themselves
while training requires the full dataset to be loaded into the memory. Significant
progress was made in 3.0 release for the GPU implementation, we will introduce the
difference between CPU and GPU in following sections.

.. note::

   Training on data from external memory is not supported by the ``exact`` tree method.

.. note::

   The feature is considered experimental but ready for public testing in 3.0. Vector-leaf
   is not yet supported.

The external memory support has gone through multiple iterations. Like the
:py:class:`~xgboost.QuantileDMatrix` with :py:class:`~xgboost.DataIter`, XGBoost loads
data batch-by-batch using a custom iterator supplied by the user. However, unlike the
:py:class:`~xgboost.QuantileDMatrix`, external memory does not concatenate the batches
unless this is explicitly specified by setting ``external_memory_concat_pages`` to true
for the GPU implementation. Instead, it will cache all batches on an external memory and
fetch them on-demand.  Go to the end of the document to see a comparison between
:py:class:`~xgboost.QuantileDMatrix` and the external memory version of
:py:class:`~xgboost.ExtMemQuantileDMatrix`.

*************
Data Iterator
*************

Starting with XGBoost 1.5, users can define their own data loader using the Python or C
interface. There are some examples in the ``demo`` directory for a quick start. To enable
external memory training, users need to define a data iterator with 2 class methods:
``next`` and ``reset``, then pass it into the :py:class:`~xgboost.DMatrix` or the
:py:class:`~xgboost.ExtMemQuantileDMatrix` constructor.

.. code-block:: python

  import os
  from typing import List, Callable
  import xgboost
  from sklearn.datasets import load_svmlight_file

  class Iterator(xgboost.DataIter):
    def __init__(self, svm_file_paths: List[str]):
      self._file_paths = svm_file_paths
      self._it = 0
      # XGBoost will generate some cache files under current directory with the prefix
      # "cache"
      super().__init__(cache_prefix=os.path.join(".", "cache"))

    def next(self, input_data: Callable):
      """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
      called by XGBoost during the construction of ``DMatrix``

      """
      if self._it == len(self._file_paths):
        # return 0 to let XGBoost know this is the end of iteration
        return 0

      # input_data is a function passed in by XGBoost who has the exact same signature of
      # ``DMatrix``
      X, y = load_svmlight_file(self._file_paths[self._it])
      input_data(data=X, label=y)
      self._it += 1
      # Return 1 to let XGBoost know we haven't seen all the files yet.
      return 1

    def reset(self):
      """Reset the iterator to its beginning"""
      self._it = 0

  it = Iterator(["file_0.svm", "file_1.svm", "file_2.svm"])

  Xy = xgboost.ExtMemQuantileDMatrix(it)
  booster = xgboost.train({"tree_method": "hist"}, Xy)

  # The ``approx`` tree method also work, but with lower performance and cannot be used
  with the quantile DMatrix.

  Xy = xgboost.DMatrix(it)
  booster = xgboost.train({"tree_method": "approx"}, Xy)

The above snippet is a simplified version of :ref:`sphx_glr_python_examples_external_memory.py`.
For an example in C, please see ``demo/c-api/external-memory/``. The iterator is the
common interface for using external memory with XGBoost, you can pass the resulting
:py:class:`DMatrix` object for training, prediction, and evaluation.

The :py:class:`ExtMemQuantileDMatrix` is an external memory version of the
:py:class:`QuantileDMatrix`. They are specifically designed for the ``hist`` tree method
for reduced memory usage and data loading overhead. See respective references for more
info.

It is important to set the batch size based on the memory available. A good starting point
is to set the batch size to 10GB per batch if you have 64GB of memory. It is *not*
recommended to set small batch sizes like 32 samples per batch, as this can severally hurt
performance in gradient boosting.

**********************************
GPU Version (GPU Hist tree method)
**********************************

External memory is supported by GPU algorithms (i.e. when ``device`` is set to
``cuda``). Starting with 3.0, the default GPU implementation is similar to what the CPU
version does. It also supports the use of :py:class:`~xgboost.ExtMemQuantileDMatrix` when
the ``hist`` tree method is employed. For a GPU device, the main memory is the device
memory, whereas the external memory can be either a disk or the CPU memory. XGBoost stages
the cache on CPU memory by default. Users can change the backing storage to disk by
specifying the ``on_host`` parameter in the :py:class:`~xgboost.DataIter`. However, using
the disk is not recommended it's likely to make the GPU slower than a CPU. The option is
here for experimental purposes only.

Inputs to the :py:class:`~xgboost.ExtMemQuantileDMatrix` (through the iterator) must be on
the GPU. This is a current limitation we aim to address in the future.

.. code-block:: python

    Xy_train = xgb.core.ExtMemQuantileDMatrix(it_train, max_bin=n_bins)
    Xy_valid = xgb.core.ExtMemQuantileDMatrix(it_valid, max_bin=n_bins, ref=Xy_train)
    booster = xgb.train(
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

In addition to the batch-based data fetching, the GPU version supports concatenating
batches into a single blob before training begins for performance reasons. For GPUs
connected via PCIe instead of nvlink, the performance overhead with batch-based training
is significant, particularly for non-dense data. Overall it can be at least five times
slower than in-core training. Concatenating pages can be used to get the performance
closer to in-core training. This option should be used in combination with subsampling to
reduce the memory usage. During concatenation, subsampling removes a portion of samples
and hence reduces the training dataset size. The GPU hist tree method supports
`gradient-based sampling`, enabling users to set a low sampling rate without compromising
accuracy. Before 3.0, concatenation with subsampling was the only option for GPU-based
external memory. After 3.0, XGBoost uses the normal batch fetching as the default.

.. code-block:: python

  param = {
    "device": "cuda",
    "external_memory_concat_pages": true,
    'subsample': 0.2,
    'sampling_method': 'gradient_based',
  }

For more information about the sampling algorithm and its use in external memory training,
see `this paper <https://arxiv.org/abs/2005.09148>`_.


**************
Best Practices
**************

In the previous section, we demonstrated how to train a tree-based model with data resided
on an external memory. This method involves iterating through data batches stored in a
cache during tree construction. For optimal performance, we recommend using the
``grow_policy=depthwise`` setting, which allows XGBoost to build an entire layer of tree
nodes with only a few batch iterations. Conversely, using the ``lossguide`` policy
requires XGBoost to iterate over the data set for each tree node, resulting in
significantly slower performance.

In addition, this ``hist`` tree method should be preferred over the ``approx`` tree method
as the former doesn't recreate the histogram bins for every iteration. Creating the
histogram bins requires loading the raw input data, which is prohibitively expensive. The
:py:class:`~xgboost.ExtMemQuantileDMatrix` designed for the ``hist`` tree method can be
used to speed up the initial data construction and the evaluation significantly for
external memory.

When external memory is used, the performance of CPU training is limited by disk IO
(input/output) speed. This means that the disk IO speed primarily determines the training
speed. Similarly, the GPU performance is limited by the PCIe bandwidth, assuming the CPU
memory is used as a cache and address translation services (ATS) is not available.

During CPU benchmarking, we used an NVMe connected to a PCIe-4 slot, other types of
storage can be too slow for practical usage. However, your system is likely to perform
some caching to reduce the overhead of the file read. See following sections for remark.

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

As one might expect, fetching data on-demand puts significant pressure on the storage
device. Today's computing device can process way more data than a storage can read in a
single unit of time. The ratio is at order of magnitudes. An GPU is capable of processing
hundred of Gigabytes of floating-point data in a split second. On the other hand, a
four-lane NVMe storage connected to a PCIe-4 slot usually has about 6GB/s of data transfer
rate. As a result, the training is likely to be severely bounded by your storage
device. Before adopting the external memory solution, some back-of-envelop calculations
might help you see whether it's viable. For instance, if your NVMe drive can transfer 4GB
(a fairly practical number) of data per second and you have a 100GB of data in compressed
XGBoost cache (which corresponds to a dense float32 numpy array with the size of 200GB,
give or take). A tree with depth 8 needs at least 16 iterations through the data when the
parameter is right. You need about 14 minutes to train a single tree without accounting
for some other overheads and assume the computation overlaps with the IO. If your dataset
happens to have TB-level size, then you might need thousands of trees to get a generalized
model. These calculations can help you get an estimate on the expected training time.

However, sometimes we can ameliorate this limitation. One should also consider that the OS
(mostly talking about the Linux kernel) can usually cache the data on host memory. It only
evicts pages when new data comes in and there's no room left. In practice, at least some
portion of the data can persist on the host memory throughout the entire training
session. We are aware of this cache when optimizing the external memory fetcher. The
compressed cache is usually smaller than the raw input data, especially when the input is
dense without any missing value. If the host memory can fit a significant portion of this
compressed cache, then the performance should be decent after initialization. Our
development so far focus on two fronts of optimization for external memory:

- Avoid iterating through the data whenever appropriate.
- If the OS can cache the data, the performance should be close to in-core training.

Starting with XGBoost 2.0, the implementation of external memory uses ``mmap``. It is not
tested against system errors like disconnected network devices (`SIGBUS`). In the face of
a bus error, you will see a hard crash and need to clean up the cache files. If the
training session might take a long time and you are using solutions like NVMe-oF, we
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
batches from external memory on-demand.  Use the :py:class:`~xgboost.QuantileDMatrix`
(with iterator if necessary) when you can fit most of your data in memory. For many
platforms, the training speed can be an order of magnitude faster than using external
memory.

****************
Text File Inputs
****************

.. warning::

   This is the original form of external memory support before 1.5 and is now deprecated,
   users are encouraged to use custom data iterator instead.

There is no big difference between using external memory version of text input and the
in-memory version of text input.  The only difference is the filename format.

The external memory version takes in the following `URI
<https://en.wikipedia.org/wiki/Uniform_Resource_Identifier>`_ format:

.. code-block:: none

  filename?format=libsvm#cacheprefix

The ``filename`` is the normal path to LIBSVM format file you want to load in, and
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
``dtrain.cache`` as an on disk cache for storing preprocessed data in an internal binary format.  For
more notes about text input formats, see :doc:`/tutorials/input_format`.

For CLI version, simply add the cache suffix, e.g. ``"../data/agaricus.txt.train?format=libsvm#dtrain.cache"``.
