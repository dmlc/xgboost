#############################
Distributed XGBoost with Dask
#############################

`Dask <https://dask.org>`_ is a parallel computing library built on Python. Dask allows
easy management of distributed workers and excels at handling large distributed data
science workflows.  The implementation in XGBoost originates from `dask-xgboost
<https://github.com/dask/dask-xgboost>`_ with some extended functionalities and a
different interface.  The tutorial here focuses on basic usage of dask with CPU tree
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

    if __name__ == "__main__":
        cluster = dask.distributed.LocalCluster()
        client = dask.distributed.Client(cluster)

        # X and y must be Dask dataframes or arrays
        num_obs = 1e5
        num_features = 20
        X = da.random.random(size=(num_obs, num_features), chunks=(1000, num_features))
        y = da.random.random(size=(num_obs, 1), chunks=(1000, 1))

        dtrain = xgb.dask.DaskDMatrix(client, X, y)

        output = xgb.dask.train(
            client,
            {"verbosity": 2, "tree_method": "hist", "objective": "reg:squarederror"},
            dtrain,
            num_boost_round=4,
            evals=[(dtrain, "train")],
        )

Here we first create a cluster in single-node mode with
:py:class:`distributed.LocalCluster`, then connect a :py:class:`distributed.Client` to
this cluster, setting up an environment for later computation.  Notice that the cluster
construction is guared by ``__name__ == "__main__"``, which is necessary otherwise there
might be obscure errors.

We then create a :py:class:`xgboost.dask.DaskDMatrix` object and pass it to
:py:func:`xgboost.dask.train`, along with some other parameters, much like XGBoost's
normal, non-dask interface. Unlike that interface, ``data`` and ``label`` must be either
:py:class:`Dask DataFrame <dask.dataframe.DataFrame>` or :py:class:`Dask Array
<dask.array.Array>` instances.

The primary difference with XGBoost's dask interface is
we pass our dask client as an additional argument for carrying out the computation. Note that if
client is set to ``None``, XGBoost will use the default client returned by dask.

There are two sets of APIs implemented in XGBoost.  The first set is functional API
illustrated in above example.  Given the data and a set of parameters, the ``train`` function
returns a model and the computation history as a Python dictionary:

.. code-block:: python

  {'booster': Booster,
   'history': dict}

For prediction, pass the ``output`` returned by ``train`` into :py:func:`xgboost.dask.predict`:

.. code-block:: python

  prediction = xgb.dask.predict(client, output, dtrain)
  # Or equivalently, pass ``output['booster']``:
  prediction = xgb.dask.predict(client, output['booster'], dtrain)

Eliminating the construction of DaskDMatrix is also possible, this can make the
computation a bit faster when meta information like ``base_margin`` is not needed:

.. code-block:: python

  prediction = xgb.dask.predict(client, output, X)
  # Use inplace version.
  prediction = xgb.dask.inplace_predict(client, output, X)

Here ``prediction`` is a dask ``Array`` object containing predictions from model if input
is a ``DaskDMatrix`` or ``da.Array``.  When putting dask collection directly into the
``predict`` function or using :py:func:`xgboost.dask.inplace_predict`, the output type
depends on input data.  See next section for details.

Alternatively, XGBoost also implements the Scikit-Learn interface with
:py:class:`~xgboost.dask.DaskXGBClassifier`, :py:class:`~xgboost.dask.DaskXGBRegressor`,
:py:class:`~xgboost.dask.DaskXGBRanker` and 2 random forest variances.  This wrapper is
similar to the single node Scikit-Learn interface in xgboost, with dask collection as
inputs and has an additional ``client`` attribute.  See following sections and
:ref:`dask-examples` for more examples.


******************
Running prediction
******************

In previous example we used ``DaskDMatrix`` as input to ``predict`` function.  In
practice, it's also possible to call ``predict`` function directly on dask collections
like ``Array`` and ``DataFrame`` and might have better prediction performance.  When
``DataFrame`` is used as prediction input, the result is a dask ``Series`` instead of
array.  Also, there's in-place predict support on dask interface, which can help reducing
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
  # where X is a dask DataFrame or dask Array containing cupy or cuDF backed data.
  prediction = xgb.dask.inplace_predict(client, booster, X)

When input is ``da.Array`` object, output is always ``da.Array``.  However, if the input
type is ``dd.DataFrame``, output can be ``dd.Series``, ``dd.DataFrame`` or ``da.Array``,
depending on output shape.  For example, when shap based prediction is used, the return
value can have 3 or 4 dimensions , in such cases an ``Array`` is always returned.

The performance of running prediction, either using ``predict`` or ``inplace_predict``, is
sensitive to number of blocks.  Internally, it's implemented using ``da.map_blocks`` and
``dd.map_partitions``.  When number of partitions is large and each of them have only
small amount of data, the overhead of calling predict becomes visible.  On the other hand,
if not using GPU, the number of threads used for prediction on each block matters.  Right
now, xgboost uses single thread for each partition.  If the number of blocks on each
workers is smaller than number of cores, then the CPU workers might not be fully utilized.

One simple optimization for running consecutive predictions is using
:py:class:`distributed.Future`:

.. code-block:: python

    dataset = [X_0, X_1, X_2]
    booster_f = client.scatter(booster, broadcast=True)
    futures = []
    for X in dataset:
        # Here we pass in a future instead of concrete booster
        shap_f = xgb.dask.predict(client, booster_f, X, pred_contribs=True)
        futures.append(shap_f)

    results = client.gather(futures)


This is only available on functional interface, as the Scikit-Learn wrapper doesn't know
how to maintain a valid future for booster.  To obtain the booster object from
Scikit-Learn wrapper object:

.. code-block:: python

    cls = xgb.dask.DaskXGBClassifier()
    cls.fit(X, y)

    booster = cls.get_booster()


**********************
Scikit-Learn interface
**********************

As mentioned previously, there's another interface that mimics the scikit-learn estimators
with higher level of of abstraction.  The interface is easier to use compared to the
functional interface but with more constraints.  It's worth mentioning that, although the
interface mimics scikit-learn estimators, it doesn't work with normal scikit-learn
utilities like ``GridSearchCV`` as scikit-learn doesn't understand distributed dask data
collection.


.. code-block:: python

    from distributed import LocalCluster, Client
    import xgboost as xgb


    def main(client: Client) -> None:
        X, y = load_data()
        clf = xgb.dask.DaskXGBClassifier(n_estimators=100, tree_method="hist")
        clf.client = client  # assign the client
        clf.fit(X, y, eval_set=[(X, y)])
        proba = clf.predict_proba(X)


    if __name__ == "__main__":
        with LocalCluster() as cluster:
            with Client(cluster) as client:
                main(client)


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
threads in each process for training.  But if ``nthread`` parameter is set:

.. code-block:: python

    output = xgb.dask.train(
        client,
        {"verbosity": 1, "nthread": 8, "tree_method": "hist"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
    )

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

        # Use ``client.compute`` instead of the ``compute`` method from dask collection
        print(await client.compute(with_m))


While for the Scikit-Learn interface, trivial methods like ``set_params`` and accessing class
attributes like ``evals_result()`` do not require ``await``.  Other methods involving
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

*****************************
Evaluation and Early Stopping
*****************************

.. versionadded:: 1.3.0

The Dask interface allows the use of validation sets that are stored in distributed collections (Dask DataFrame or Dask Array). These can be used for evaluation and early stopping.

To enable early stopping, pass one or more validation sets containing ``DaskDMatrix`` objects.

.. code-block:: python

    import dask.array as da
    import xgboost as xgb

    num_rows = 1e6
    num_features = 100
    num_partitions = 10
    rows_per_chunk = num_rows / num_partitions

    data = da.random.random(
        size=(num_rows, num_features),
        chunks=(rows_per_chunk, num_features)
    )

    labels = da.random.random(
        size=(num_rows, 1),
        chunks=(rows_per_chunk, 1)
    )

    X_eval = da.random.random(
        size=(num_rows, num_features),
        chunks=(rows_per_chunk, num_features)
    )

    y_eval = da.random.random(
        size=(num_rows, 1),
        chunks=(rows_per_chunk, 1)
    )

    dtrain = xgb.dask.DaskDMatrix(
        client=client,
        data=data,
        label=labels
    )

    dvalid = xgb.dask.DaskDMatrix(
        client=client,
        data=X_eval,
        label=y_eval
    )

    result = xgb.dask.train(
        client=client,
        params={
            "objective": "reg:squarederror",
        },
        dtrain=dtrain,
        num_boost_round=10,
        evals=[(dvalid, "valid1")],
        early_stopping_rounds=3
    )

When validation sets are provided to ``xgb.dask.train()`` in this way, the model object returned by ``xgb.dask.train()`` contains a history of evaluation metrics for each validation set, across all boosting rounds.

.. code-block:: python

    print(result["history"])
    # {'valid1': OrderedDict([('rmse', [0.28857, 0.28858, 0.288592, 0.288598])])}

If early stopping is enabled by also passing ``early_stopping_rounds``, you can check the best iteration in the returned booster.

.. code-block:: python

    booster = result["booster"]
    print(booster.best_iteration)
    best_model = booster[: booster.best_iteration]


*******************
Other customization
*******************

XGBoost dask interface accepts other advanced features found in single node Python
interface, including callback functions, custom evaluation metric and objective:

.. code-block:: python

    def eval_error_metric(predt, dtrain: xgb.DMatrix):
        label = dtrain.get_label()
        r = np.zeros(predt.shape)
        gt = predt > 0.5
        r[gt] = 1 - label[gt]
        le = predt <= 0.5
        r[le] = label[le]
        return 'CustomErr', np.sum(r)

    # custom callback
    early_stop = xgb.callback.EarlyStopping(
        rounds=early_stopping_rounds,
        metric_name="CustomErr",
        data_name="Train",
        save_best=True,
    )

    booster = xgb.dask.train(
        client,
        params={
            "objective": "binary:logistic",
            "eval_metric": ["error", "rmse"],
            "tree_method": "hist",
        },
        dtrain=D_train,
        evals=[(D_train, "Train"), (D_valid, "Valid")],
        feval=eval_error_metric,  # custom evaluation metric
        num_boost_round=100,
        callbacks=[early_stop],
    )


.. _tracker-ip:

***************
Troubleshooting
***************

.. versionadded:: 1.6.0

In some environments XGBoost might fail to resolve the IP address of the scheduler, a
symptom is user receiving ``OSError: [Errno 99] Cannot assign requested address`` error
during training.  A quick workaround is to specify the address explicitly.  To do that
dask config is used:

.. code-block:: python

    import dask
    from distributed import Client
    from xgboost import dask as dxgb
    # let xgboost know the scheduler address
    dask.config.set({"xgboost.scheduler_address": "192.0.0.100"})

    with Client(scheduler_file="sched.json") as client:
        reg = dxgb.DaskXGBRegressor()

    # or we can specify the port too
    with dask.config.set({"xgboost.scheduler_address": "192.0.0.100:12345"}):
        reg = dxgb.DaskXGBRegressor()


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
  :py:class:`xgboost.dask.DaskDeviceQuantileDMatrix` as a drop in replacement for ``DaskDMatrix``
  to reduce overall memory usage.  See
  :ref:`sphx_glr_python_dask-examples_gpu_training.py` for an example.

- Use in-place prediction when possible.

References:

#. https://github.com/dask/dask/issues/6833
#. https://stackoverflow.com/questions/45941528/how-to-efficiently-send-a-large-numpy-array-to-the-cluster-with-dask-array
