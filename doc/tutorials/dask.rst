#############################
Distributed XGBoost with Dask
#############################

`Dask <https://dask.org>`_ is a parallel computing library built on Python. Dask allows
easy management of distributed workers and excels handling large distributed data science
workflows.  The implementation in XGBoost originates from `dask-xgboost
<https://github.com/dask/dask-xgboost>`_ with some extended functionalities and a
different interface.  Right now it is still under construction and may change (with proper
warnings) in the future.

************
Requirements
************

Dask is trivial to install using either pip or conda.  `See here for official install
documentation <https://docs.dask.org/en/latest/install.html>`_.  For accelerating XGBoost
with GPU, `dask-cuda <https://github.com/rapidsai/dask-cuda>`_ is recommended for creating
GPU clusters.


********
Overview
********

There are 3 different components in dask from a user's perspective, namely a scheduler,
bunch of workers and some clients connecting to the scheduler.  For using XGBoost with
dask, one needs to call XGBoost dask interface from the client side.  A small example
illustrates the basic usage:

.. code-block:: python

  cluster = LocalCluster(n_workers=4, threads_per_worker=1)
  client = Client(cluster)

  dtrain = xgb.dask.DaskDMatrix(client, X, y)  # X and y are dask dataframes or arrays

  output = xgb.dask.train(client,
                          {'verbosity': 2,
                           'tree_method': 'hist'},
                          dtrain,
                          num_boost_round=4, evals=[(dtrain, 'train')])

Here we first create a cluster in single-node mode wtih ``distributed.LocalCluster``, then
connect a ``client`` to this cluster, setting up environment for later computation.
Similar to non-distributed interface, we create a ``DMatrix`` object and pass it to
``train`` along with some other parameters.  Except in dask interface, client is an extra
argument for carrying out the computation, when set to ``None`` XGBoost will use the
default client returned from dask.

There are two sets of APIs implemented in XGBoost.  The first set is functional API
illustrated in above example.  Given the data and a set of parameters, `train` function
returns a model and the computation history as Python dictionary

.. code-block:: python

  {'booster': Booster,
   'history': dict}

For prediction, pass the ``output`` returned by ``train`` into ``xgb.dask.predict``

.. code-block:: python

  prediction = xgb.dask.predict(client, output, dtrain)

Or equivalently, pass ``output['booster']``:

.. code-block:: python

  prediction = xgb.dask.predict(client, output['booster'], dtrain)

Here ``prediction`` is a dask ``Array`` object containing predictions from model.

Another set of API is a Scikit-Learn wrapper, which mimics the stateful Scikit-Learn
interface with ``DaskXGBClassifier`` and ``DaskXGBRegressor``.  See ``xgboost/demo/dask``
for more examples.

*******
Threads
*******

XGBoost has built in support for parallel computation through threads by the setting
``nthread`` parameter (``n_jobs`` for scikit-learn).  If these parameters are set, they
will override the configuration in Dask.  For example:

.. code-block:: python

  with LocalCluster(n_workers=7, threads_per_worker=4) as cluster:

There are 4 threads allocated for each dask worker.  Then by default XGBoost will use 4
threads in each process for both training and prediction.  But if ``nthread`` parameter is
set:

.. code-block:: python

  output = xgb.dask.train(client,
                          {'verbosity': 1,
                           'nthread': 8,
                           'tree_method': 'hist'},
                          dtrain,
                          num_boost_round=4, evals=[(dtrain, 'train')])

XGBoost will use 8 threads in each training process.

*****************************************************************************
Why is the initialization of ``DaskDMatrix``  so slow and throws weird errors
*****************************************************************************

The dask API in XGBoost requires construction of ``DaskDMatrix``.  With ``Scikit-Learn``
interface, ``DaskDMatrix`` is implicitly constructed for each input data during `fit` or
`predict`.  You might have observed its construction is taking incredible amount of time,
and sometimes throws error that doesn't seem to be relevant to `DaskDMatrix`.  Here is a
brief explanation for why.  By default most of dask's computation is `lazy
<https://docs.dask.org/en/latest/user-interfaces.html#laziness-and-computing>`_, which
means the computation is not carried out until you explicitly ask for result, either by
calling `compute()` or `wait()`.  See above link for details in dask, and `this wiki
<https://en.wikipedia.org/wiki/Lazy_evaluation>`_ for general concept of lazy evaluation.
The `DaskDMatrix` constructor forces all lazy computation to materialize, which means it's
where all your earlier computation actually being carried out, including operations like
`dd.read_csv()`.  To isolate the computation in `DaskDMatrix` from other lazy
computations, one can explicitly wait for results of input data before calling constructor
of `DaskDMatrix`.  Also dask's `web interface
<https://distributed.dask.org/en/latest/web.html>`_ can be used to monitor what operations
are currently being performed.

***********
Limitations
***********

Basic functionalities including training and generating predictions for regression and
classification are implemented.  But there are still some other limitations we haven't
addressed yet.

- Label encoding for Scikit-Learn classifier may not be supported.  Meaning that user need
  to encode their training labels into discrete values first.
- Ranking is not supported right now.
- Empty worker is not well supported by classifier.  If the training hangs for classifier
  with a warning about empty DMatrix, please consider balancing your data first.  But
  regressor works fine with empty DMatrix.
- Callback functions are not tested.
- Only ``GridSearchCV`` from ``scikit-learn`` is supported for dask interface.  Meaning
  that we can distribute data among workers but have to train one model at a time.  If you
  want to scale up grid searching with model parallelism by ``dask-ml``, please consider
  using normal ``scikit-learn`` interface like `xgboost.XGBRegressor` for now.
