#####################################
Using XGBoost External Memory Version
#####################################

When working with large datasets, training XGBoost models can be challenging as the entire
dataset needs to be loaded into memory. This can be costly and sometimes
infeasible. Staring from 1.5, users can define a custom iterator to load data in chunks
for running XGBoost algorithms. External memory can be used for both training and
prediction, but training is the primary use case and it will be our focus in this
tutorial. For prediction and evaluation, users can iterate through the data themselves
while training requires the full dataset to be loaded into the memory.

During training, there are two different modes for external memory support available in
XGBoost, one for CPU-based algorithms like ``hist`` and ``approx``, another one for the
GPU-based training algorithm. We will introduce them in the following sections.

.. note::

   Training on data from external memory is not supported by the ``exact`` tree method.

.. note::

   The feature is still experimental as of 2.0. The performance is not well optimized.

The external memory support has gone through multiple iterations and is still under heavy
development. Like the :py:class:`~xgboost.QuantileDMatrix` with
:py:class:`~xgboost.DataIter`, XGBoost loads data batch-by-batch using a custom iterator
supplied by the user. However, unlike the :py:class:`~xgboost.QuantileDMatrix`, external
memory will not concatenate the batches unless GPU is used (it uses a hybrid approach,
more details follow). Instead, it will cache all batches on the external memory and fetch
them on-demand.  Go to the end of the document to see a comparison between
:py:class:`~xgboost.QuantileDMatrix` and external memory.

*************
Data Iterator
*************

Starting from XGBoost 1.5, users can define their own data loader using Python or C
interface.  There are some examples in the ``demo`` directory for quick start.  This is a
generalized version of text input external memory, where users no longer need to prepare a
text file that XGBoost recognizes.  To enable the feature, users need to define a data
iterator with 2 class methods: ``next`` and ``reset``, then pass it into the
:py:class:`~xgboost.DMatrix` constructor.

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
  Xy = xgboost.DMatrix(it)

  # The ``approx`` also work, but with low performance. GPU implementation is different from CPU.
  # as noted in following sections.
  booster = xgboost.train({"tree_method": "hist"}, Xy)


The above snippet is a simplified version of :ref:`sphx_glr_python_examples_external_memory.py`.
For an example in C, please see ``demo/c-api/external-memory/``. The iterator is the
common interface for using external memory with XGBoost, you can pass the resulting
:py:class:`DMatrix` object for training, prediction, and evaluation.

It is important to set the batch size based on the memory available. A good starting point
is to set the batch size to 10GB per batch if you have 64GB of memory. It is *not*
recommended to set small batch sizes like 32 samples per batch, as this can seriously hurt
performance in gradient boosting.

***********
CPU Version
***********

In the previous section, we demonstrated how to train a tree-based model using the
``hist`` tree method on a CPU. This method involves iterating through data batches stored
in a cache during tree construction. For optimal performance, we recommend using the
``grow_policy=depthwise`` setting, which allows XGBoost to build an entire layer of tree
nodes with only a few batch iterations. Conversely, using the ``lossguide`` policy
requires XGBoost to iterate over the data set for each tree node, resulting in slower
performance.

If external memory is used, the performance of CPU training is limited by IO
(input/output) speed. This means that the disk IO speed primarily determines the training
speed. During benchmarking, we used an NVMe connected to a PCIe-4 slot, other types of
storage can be too slow for practical usage. In addition, your system may perform caching
to reduce the overhead of file reading.

**********************************
GPU Version (GPU Hist tree method)
**********************************

External memory is supported by GPU algorithms (i.e. when ``device`` is set to
``cuda``). However, the algorithm used for GPU is different from the one used for
CPU. When training on a CPU, the tree method iterates through all batches from external
memory for each step of the tree construction algorithm. On the other hand, the GPU
algorithm uses a hybrid approach. It iterates through the data during the beginning of
each iteration and concatenates all batches into one in GPU memory for performance
reasons. To reduce overall memory usage, users can utilize subsampling. The GPU hist tree
method supports `gradient-based sampling`, enabling users to set a low sampling rate
without compromising accuracy.

.. code-block:: python

  param = {
    ...
    'subsample': 0.2,
    'sampling_method': 'gradient_based',
  }

For more information about the sampling algorithm and its use in external memory training,
see `this paper <https://arxiv.org/abs/2005.09148>`_.

.. warning::

   When GPU is running out of memory during iteration on external memory, user might
   receive a segfault instead of an OOM exception.

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
if it's passed to :py:class:`~xgboost.DMatrix`, it instead enables the external memory
feature. The :py:class:`~xgboost.QuantileDMatrix` concatenates the data on memory after
compression and doesn't fetch data during training. On the other hand, the external memory
:py:class:`~xgboost.DMatrix` fetches data batches from external memory on-demand.  Use the
:py:class:`~xgboost.QuantileDMatrix` (with iterator if necessary) when you can fit most of
your data in memory. The training would be an order of magnitude faster than using
external memory.

****************
Text File Inputs
****************

.. warning::

   This is the original form of external memory support before 1.5, users are encouraged
   to use custom data iterator instead.

There is no big difference between using external memory version of text input and the
in-memory version.  The only difference is the filename format.

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
