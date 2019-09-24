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
                           'nthread': 1,
                           'tree_method': 'hist'},
                          dtrain,
                          num_boost_round=4, evals=[(dtrain, 'train')])

Here we first create a cluster in signle-node mode wtih ``distributed.LocalCluster``, then
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


***********
Limitations
***********

Basic functionalities including training and generating predictions for regression and
classification are implemented.  But there are still some other limitations we haven't
addressed yet.

- Label encoding for Scikit-Learn classifier.
- Ranking
- Callback functions are not tested.
- To use cross validation one needs to explicitly train different models instead of using
  a functional API like ``xgboost.cv``.
