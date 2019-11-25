############################################
Using XGBoost External Memory Version (beta)
############################################
There is no big difference between using external memory version and in-memory version.
The only difference is the filename format.

The external memory version takes in the following `URI <https://en.wikipedia.org/wiki/Uniform_Resource_Identifier>`_ format:

.. code-block:: none

  filename#cacheprefix

The ``filename`` is the normal path to libsvm format file you want to load in, and
``cacheprefix`` is a path to a cache file that XGBoost will use for caching preprocessed
data in binary form.

.. note:: External memory is also available with GPU algorithms (i.e. when ``tree_method`` is set to ``gpu_hist``)

To provide a simple example for illustration, extracting the code from
`demo/guide-python/external_memory.py <https://github.com/dmlc/xgboost/blob/master/demo/guide-python/external_memory.py>`_. If
you have a dataset stored in a file similar to ``agaricus.txt.train`` with libSVM format, the external memory support can be enabled by:

.. code-block:: python

  dtrain = DMatrix('../data/agaricus.txt.train#dtrain.cache')

XGBoost will first load ``agaricus.txt.train`` in, preprocess it, then write to a new file named
``dtrain.cache`` as an on disk cache for storing preprocessed data in a internal binary format.  For
more notes about text input formats, see :doc:`/tutorials/input_format`.

.. code-block:: python

  dtrain = xgb.DMatrix('../data/agaricus.txt.train#dtrain.cache')

For CLI version, simply add the cache suffix, e.g. ``"../data/agaricus.txt.train#dtrain.cache"``.

****************
Performance Note
****************
* the parameter ``nthread`` should be set to number of **physical** cores

  - Most modern CPUs use hyperthreading, which means a 4 core CPU may carry 8 threads
  - Set ``nthread`` to be 4 for maximum performance in such case

*******************
Distributed Version
*******************
The external memory mode naturally works on distributed version, you can simply set path like

.. code-block:: none

  data = "hdfs://path-to-data/#dtrain.cache"

XGBoost will cache the data to the local position. When you run on YARN, the current folder is temporal
so that you can directly use ``dtrain.cache`` to cache to current folder.

**********
Usage Note
**********
* This is an experimental version
* Currently only importing from libsvm format is supported
* OSX is not tested.

  - Contribution of ingestion from other common external memory data source is welcomed
