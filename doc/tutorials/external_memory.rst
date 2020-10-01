#####################################
Using XGBoost External Memory Version
#####################################
There is no big difference between using external memory version and in-memory version.
The only difference is the filename format.

The external memory version takes in the following `URI <https://en.wikipedia.org/wiki/Uniform_Resource_Identifier>`_ format:

.. code-block:: none

  filename#cacheprefix

The ``filename`` is the normal path to libsvm format file you want to load in, and
``cacheprefix`` is a path to a cache file that XGBoost will use for caching preprocessed
data in binary form.

To load from csv files, use the following syntax:

.. code-block:: none

  filename.csv?format=csv&label_column=0#cacheprefix

where ``label_column`` should point to the csv column acting as the label.

To provide a simple example for illustration, extracting the code from
`demo/guide-python/external_memory.py <https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py>`_. If
you have a dataset stored in a file similar to ``agaricus.txt.train`` with libSVM format, the external memory support can be enabled by:

.. code-block:: python

  dtrain = DMatrix('../data/agaricus.txt.train#dtrain.cache')

XGBoost will first load ``agaricus.txt.train`` in, preprocess it, then write to a new file named
``dtrain.cache`` as an on disk cache for storing preprocessed data in an internal binary format.  For
more notes about text input formats, see :doc:`/tutorials/input_format`.

For CLI version, simply add the cache suffix, e.g. ``"../data/agaricus.txt.train#dtrain.cache"``.

***********
GPU Version
***********
External memory is fully supported in GPU algorithms (i.e. when ``tree_method`` is set to ``gpu_hist``).

If you are still getting out-of-memory errors after enabling external memory, try subsampling the
data to further reduce GPU memory usage:

.. code-block:: python

  param = {
    ...
    'subsample': 0.1,
    'sampling_method': 'gradient_based',
  }

For more information, see `this paper <https://arxiv.org/abs/2005.09148>`_.

*******************
Distributed Version
*******************
The external memory mode naturally works on distributed version, you can simply set path like

.. code-block:: none

  data = "hdfs://path-to-data/#dtrain.cache"

XGBoost will cache the data to the local position. When you run on YARN, the current folder is temporary
so that you can directly use ``dtrain.cache`` to cache to current folder.

***********
Limitations
***********
* The ``hist`` tree method hasn't been tested thoroughly with external memory support (see
  `this issue <https://github.com/dmlc/xgboost/issues/4093>`_).
* OSX is not tested.
