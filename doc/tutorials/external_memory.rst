#####################################
Using XGBoost External Memory Version
#####################################

XGBoost supports loading data from external memory using builtin data parser.  And
starting from version 1.5, users can also define a custom iterator to load data in chunks.
The feature is still experimental and not yet ready for production use.  In this tutorial
we will introduce both methods.  Please note that training on data from external memory is
not supported by ``exact`` tree method.

*************
Data Iterator
*************

Starting from XGBoost 1.5, users can define their own data loader using Python or C
interface.  There are some examples in the ``demo`` directory for quick start.  This is a
generalized version of text input external memory, where users no longer need to prepare a
text file that XGBoost recognizes.  To enable the feature, user need to define a data
iterator with 2 class methods ``next`` and ``reset`` then pass it into ``DMatrix``
constructor.

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

  # Other tree methods including ``hist`` and ``gpu_hist`` also work, but has some caveats
  # as noted in following sections.
  booster = xgboost.train({"tree_method": "approx"}, Xy)


The above snippet is a simplified version of ``demo/guide-python/external_memory.py``.  For
an example in C, please see ``demo/c-api/external-memory/``.

****************
Text File Inputs
****************

There is no big difference between using external memory version and in-memory version.
The only difference is the filename format.

The external memory version takes in the following `URI <https://en.wikipedia.org/wiki/Uniform_Resource_Identifier>`_ format:

.. code-block:: none

  filename?format=libsvm#cacheprefix

The ``filename`` is the normal path to LIBSVM format file you want to load in, and
``cacheprefix`` is a path to a cache file that XGBoost will use for caching preprocessed
data in binary form.

To load from csv files, use the following syntax:

.. code-block:: none

  filename.csv?format=csv&label_column=0#cacheprefix

where ``label_column`` should point to the csv column acting as the label.

To provide a simple example for illustration, extracting the code from
`demo/guide-python/external_memory.py <https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py>`_. If
you have a dataset stored in a file similar to ``agaricus.txt.train`` with LIBSVM format, the external memory support can be enabled by:

.. code-block:: python

  dtrain = DMatrix('../data/agaricus.txt.train?format=libsvm#dtrain.cache')

XGBoost will first load ``agaricus.txt.train`` in, preprocess it, then write to a new file named
``dtrain.cache`` as an on disk cache for storing preprocessed data in an internal binary format.  For
more notes about text input formats, see :doc:`/tutorials/input_format`.

For CLI version, simply add the cache suffix, e.g. ``"../data/agaricus.txt.train?format=libsvm#dtrain.cache"``.


**********************************
GPU Version (GPU Hist tree method)
**********************************
External memory is supported in GPU algorithms (i.e. when ``tree_method`` is set to ``gpu_hist``).

If you are still getting out-of-memory errors after enabling external memory, try subsampling the
data to further reduce GPU memory usage:

.. code-block:: python

  param = {
    ...
    'subsample': 0.1,
    'sampling_method': 'gradient_based',
  }

For more information, see `this paper <https://arxiv.org/abs/2005.09148>`_.  Internally
the tree method still concatenate all the chunks into 1 final histogram index due to
performance reason, but in compressed format.  So its scalability has an upper bound but
still has lower memory cost in general.

***********
CPU Version
***********

For CPU histogram based tree methods (``approx``, ``hist``) it's recommended to use
``grow_policy=depthwise`` for performance reason.  Iterating over data batches is slow,
with ``depthwise`` policy XGBoost can build a entire layer of tree nodes with a few
iterations, while with ``lossguide`` XGBoost needs to iterate over the data set for each
tree node.
