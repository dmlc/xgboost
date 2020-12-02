#############################
Distributed XGBoost with Dask
#############################

`Dask <https://dask.org>`_ is a parallel computing library built on Python. Dask allows
easy management of distributed workers and excels at handling large distributed data science
workflows.  The implementation in XGBoost originates from `dask-xgboost
<https://github.com/dask/dask-xgboost>`_ with some extended functionalities and a
different interface.  Right now it is still under construction and may change (with proper
warnings) in the future.  The tutorial here focuses on basic usage of dask with CPU tree
algorithms.  For an overview of GPU based training and internal workings, see `A New,
Official Dask API for XGBoost
<https://medium.com/rapids-ai/a-new-official-dask-api-for-xgboost-e8b10f3d1eb7>`_.

**Contents**

.. contents::
  :backlinks: none
  :local:

************
Requirements
************

Dask can be installed using either pip or conda (see the dask `installation
documentation <https://docs.dask.org/en/latest/install.html>`_ for more information).  For
accelerating XGBoost with GPUs, `dask-cuda <https://github.com/rapidsai/dask-cuda>`_ is
recommended for creating GPU clusters.


********
Overview
********

A dask cluster consists of three different components: a centralized scheduler, one or
more workers, and one or more clients which act as the user-facing entry point for submitting
tasks to the cluster.  When using XGBoost with dask, one needs to call the XGBoost dask interface
from the client side.  Below is a small example which illustrates basic usage of running XGBoost
on a dask cluster:

.. code-block:: python

  import xgboost as xgb
  import dask.array as da
  import dask.distributed

  cluster = dask.distributed.LocalCluster(n_workers=4, threads_per_worker=1)
  client = dask.distributed.Client(cluster)

  # X and y must be Dask dataframes or arrays
  num_obs = 1e5
  num_features = 20
  X = da.random.random(
      size=(num_obs, num_features)
  )
  y = da.random.choice(
      a=[0, 1],
      size=num_obs,
      replace=True
  )

  dtrain = xgb.dask.DaskDMatrix(client, X, y)

  output = xgb.dask.train(client,
                          {'verbosity': 2,
                           'tree_method': 'hist',
                           'objective': 'binary:logistic'
                           },
                          dtrain,
                          num_boost_round=4, evals=[(dtrain, 'train')])

Here we first create a cluster in single-node mode with ``dask.distributed.LocalCluster``, then
connect a ``dask.distributed.Client`` to this cluster, setting up an environment for later computation.

We then create a ``DaskDMatrix`` object and pass it to ``train``, along with some other parameters,
much like XGBoost's normal, non-dask interface. Unlike that interface, ``data`` and ``label`` must
be either `Dask DataFrame <https://examples.dask.org/dataframe.html>`_ or
`Dask Array <https://examples.dask.org/array.html>`_ instances.

The primary difference with XGBoost's dask interface is
we pass our dask client as an additional argument for carrying out the computation. Note that if
client is set to ``None``, XGBoost will use the default client returned by dask.

There are two sets of APIs implemented in XGBoost.  The first set is functional API
illustrated in above example.  Given the data and a set of parameters, the ``train`` function
returns a model and the computation history as a Python dictionary:

.. code-block:: python

  {'booster': Booster,
   'history': dict}

For prediction, pass the ``output`` returned by ``train`` into ``xgb.dask.predict``:

.. code-block:: python

  prediction = xgb.dask.predict(client, output, dtrain)

Or equivalently, pass ``output['booster']``:

.. code-block:: python

  prediction = xgb.dask.predict(client, output['booster'], dtrain)

Here ``prediction`` is a dask ``Array`` object containing predictions from model.

Alternatively, XGBoost also implements the Scikit-Learn interface with ``DaskXGBClassifier``
and ``DaskXGBRegressor``. See ``xgboost/demo/dask`` for more examples.


******************
Running prediction
******************

In previous example we used ``DaskDMatrix`` as input to ``predict`` function.  In
practice, it's also possible to call ``predict`` function directly on dask collections
like ``Array`` and ``DataFrame`` and might have better prediction performance.  When
``DataFrame`` is used as prediction input, the result is a dask ``Series`` instead of
array.  Also, there's inplace predict support on dask interface, which can help reducing
both memory usage and prediction time.

.. code-block:: python

  # dtrain is the DaskDMatrix defined above.
  prediction = xgb.dask.predict(client, booster, dtrain)

or equivalently:

.. code-block:: python

  # where X is a dask DataFrame or dask Array.
  prediction = xgb.dask.predict(client, booster, X)

Also for inplace prediction:

.. code-block:: python

  booster.set_param({'predictor': 'gpu_predictor'})
  # where X is a dask DataFrame or dask Array.
  prediction = xgb.dask.inplace_predict(client, booster, X)


***************************
Working with other clusters
***************************

``LocalCluster`` is mostly used for testing.  In real world applications some other
clusters might be preferred.  Examples are like ``LocalCUDACluster`` for single node
multi-GPU instance, manually launched cluster by using command line utilities like
``dask-worker`` from ``distributed`` for not yet automated environments.  Some special
clusters like ``KubeCluster`` from ``dask-kubernetes`` package are also possible.  The
dask API in xgboost is orthogonal to the cluster type and can be used with any of them.  A
typical testing workflow with ``KubeCluster`` looks like this:

.. code-block:: python

  from dask_kubernetes import KubeCluster  # Need to install the ``dask-kubernetes`` package
  from dask.distributed import Client
  import xgboost as xgb
  import dask
  import dask.array as da

  dask.config.set({"kubernetes.scheduler-service-type": "LoadBalancer",
                   "kubernetes.scheduler-service-wait-timeout": 360,
                   "distributed.comm.timeouts.connect": 360})


  def main():
      '''Connect to a remote kube cluster with GPU nodes and run training on it.'''
      m = 1000
      n = 10
      kWorkers = 2                # assuming you have 2 GPU nodes on that cluster.
      # You need to work out the worker-spec youself.  See document in dask_kubernetes for
      # its usage.  Here we just want to show that XGBoost works on various clusters.
      cluster = KubeCluster.from_yaml('worker-spec.yaml', deploy_mode='remote')
      cluster.scale(kWorkers)     # scale to use all GPUs

      with Client(cluster) as client:
          X = da.random.random(size=(m, n), chunks=100)
          y = da.random.random(size=(m, ), chunks=100)

          regressor = xgb.dask.DaskXGBRegressor(n_estimators=10, missing=0.0)
          regressor.client = client
          regressor.set_params(tree_method='gpu_hist')
          regressor.fit(X, y, eval_set=[(X, y)])


  if __name__ == '__main__':
      # Launch the kube cluster on somewhere like GKE, then run this as client process.
      # main function will connect to that cluster and start training xgboost model.
      main()


However, these clusters might have their subtle differences like network configuration, or
specific cluster implementation might contains bugs that we are not aware of.  Open an
issue if such case is found and there's no documentation on how to resolve it in that
cluster implementation.

*******
Threads
*******

XGBoost has built in support for parallel computation through threads by the setting
``nthread`` parameter (``n_jobs`` for scikit-learn).  If these parameters are set, they
will override the configuration in Dask.  For example:

.. code-block:: python

  with dask.distributed.LocalCluster(n_workers=7, threads_per_worker=4) as cluster:

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

********************
Working with asyncio
********************

.. versionadded:: 1.2.0

XGBoost's dask interface supports the new ``asyncio`` in Python and can be integrated into
asynchronous workflows.  For using dask with asynchronous operations, please refer to
`this dask example <https://examples.dask.org/applications/async-await.html>`_ and document in
`distributed <https://distributed.dask.org/en/latest/asynchronous.html>`_. To use XGBoost's
dask interface asynchronously, the ``client`` which is passed as an argument for training and
prediction must be operating in asynchronous mode by specifying ``asynchronous=True`` when the
``client`` is created (example below). All functions (including ``DaskDMatrix``) provided
by the functional interface will then return coroutines which can then be awaited to retrieve
their result.

Functional interface:

.. code-block:: python

    async with dask.distributed.Client(scheduler_address, asynchronous=True) as client:
        X, y = generate_array()
        m = await xgb.dask.DaskDMatrix(client, X, y)
        output = await xgb.dask.train(client, {}, dtrain=m)

        with_m = await xgb.dask.predict(client, output, m)
        with_X = await xgb.dask.predict(client, output, X)
        inplace = await xgb.dask.inplace_predict(client, output, X)

        # Use `client.compute` instead of the `compute` method from dask collection
        print(await client.compute(with_m))


While for the Scikit-Learn interface, trivial methods like ``set_params`` and accessing class
attributes like ``evals_result_`` do not require ``await``.  Other methods involving
actual computation will return a coroutine and hence require awaiting:

.. code-block:: python

    async with dask.distributed.Client(scheduler_address, asynchronous=True) as client:
        X, y = generate_array()
        regressor = await xgb.dask.DaskXGBRegressor(verbosity=1, n_estimators=2)
        regressor.set_params(tree_method='hist')  # trivial method, synchronous operation
        regressor.client = client  #  accessing attribute, synchronous operation
        regressor = await regressor.fit(X, y, eval_set=[(X, y)])
        prediction = await regressor.predict(X)

        # Use `client.compute` instead of the `compute` method from dask collection
        print(await client.compute(prediction))

*****************************************************************************
Why is the initialization of ``DaskDMatrix``  so slow and throws weird errors
*****************************************************************************

The dask API in XGBoost requires construction of ``DaskDMatrix``.  With the Scikit-Learn
interface, ``DaskDMatrix`` is implicitly constructed for all input data during the ``fit`` or
``predict`` steps.  You might have observed that ``DaskDMatrix`` construction can take large amounts of time,
and sometimes throws errors that don't seem to be relevant to ``DaskDMatrix``.  Here is a
brief explanation for why.  By default most dask computations are `lazily evaluated
<https://docs.dask.org/en/latest/user-interfaces.html#laziness-and-computing>`_, which
means that computation is not carried out until you explicitly ask for a result by, for example,
calling ``compute()``.  See the previous link for details in dask, and `this wiki
<https://en.wikipedia.org/wiki/Lazy_evaluation>`_ for information on the general concept of lazy evaluation.
The ``DaskDMatrix`` constructor forces lazy computations to be evaluated, which means it's
where all your earlier computation actually being carried out, including operations like
``dd.read_csv()``.  To isolate the computation in ``DaskDMatrix`` from other lazy
computations, one can explicitly wait for results of input data before constructing a ``DaskDMatrix``.
Also dask's `diagnostics dashboard <https://distributed.dask.org/en/latest/web.html>`_ can be used to
monitor what operations are currently being performed.

************
Memory Usage
************

Here are some pratices on reducing memory usage with dask and xgboost.

- In a distributed work flow, data is best loaded by dask collections directly instead of
  loaded by client process.  When loading with client process is unavoidable, use
  ``client.scatter`` to distribute data from client process to workers.  See [2] for a
  nice summary.

- When using GPU input, like dataframe loaded by ``dask_cudf``, you can try
  ``xgboost.dask.DaskDeviceQuantileDMatrix`` as a drop in replacement for ``DaskDMatrix``
  to reduce overall memory usage.  See ``demo/dask/gpu_training.py`` for an example.

- Use inplace prediction when possible.

References:

#. https://github.com/dask/dask/issues/6833
#. https://stackoverflow.com/questions/45941528/how-to-efficiently-send-a-large-numpy-array-to-the-cluster-with-dask-array

***********
Limitations
***********

Basic functionality including model training and generating classification and regression predictions
have been implemented.  However, there are still some other limitations we haven't
addressed yet:

- Label encoding for the ``DaskXGBClassifier`` classifier may not be supported.  So users need
  to encode their training labels into discrete values first.
- Ranking is not yet supported.
- Callback functions are not tested.
