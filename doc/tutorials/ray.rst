############################
Distributed XGBoost with Ray
############################

`Ray <https://ray.io/>`_ is a general purpose distributed execution framework.
Ray can be used to scale computations from a single node to a cluster of hundreds
of nodes without changing any code.

The Python bindings of Ray come with a collection of well maintained
machine learning libraries for hyperparameter optimization and model serving.

The `XGBoost-Ray <https://github.com/ray-project/xgboost_ray>`_ project provides
an interface to run XGBoost training and prediction jobs on a Ray cluster. It allows
to utilize distributed data representations, such as
`Modin <https://modin.readthedocs.io/en/latest/>`_ dataframes,
as well as distributed loading from cloud storage (e.g. Parquet files).

XGBoost-Ray integrates well with hyperparameter optimization library Ray Tune, and
implements advanced fault tolerance handling mechanisms. With Ray you can scale
your training jobs to hundreds of nodes just by adding new
nodes to a cluster. You can also use Ray to leverage multi GPU XGBoost training.

Installing and starting Ray
===========================
Ray can be installed from PyPI like this:

.. code-block:: bash

    pip install ray

If you're using Ray on a single machine, you don't need to do anything else -
XGBoost-Ray will automatically start a local Ray cluster when used.

If you want to use Ray on a cluster, you can use the
`Ray cluster launcher <https://docs.ray.io/en/master/cluster/cloud.html>`_.

Installing XGBoost-Ray
======================
XGBoost-Ray is also available via PyPI:

.. code-block:: bash

    pip install xgboost_ray

This will install all dependencies needed to run XGBoost on Ray, including
Ray itself if it hasn't been installed before.

Using XGBoost-Ray for training and prediction
=============================================
XGBoost-Ray uses the same API as core XGBoost. There are only two differences:

1. Instead of using a ``xgboost.DMatrix``, you'll use a ``xgboost_ray.RayDMatrix`` object
2. There is an additional :class:`ray_params <xgboost_ray.RayParams>` parameter that you can use to configure distributed training.

Simple training example
-----------------------

To run this simple example, you'll need to install
`scikit-learn <https://scikit-learn.org/>`_ (with ``pip install sklearn``).

In this example, we will load the `breast cancer dataset <https://archive.ics.uci.edu/ml/datasets/breast+cancer>`_
and train a binary classifier using two actors.

.. code-block:: python

    from xgboost_ray import RayDMatrix, RayParams, train
    from sklearn.datasets import load_breast_cancer

    train_x, train_y = load_breast_cancer(return_X_y=True)
    train_set = RayDMatrix(train_x, train_y)

    evals_result = {}
    bst = train(
        {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
        },
        train_set,
        evals_result=evals_result,
        evals=[(train_set, "train")],
        verbose_eval=False,
        ray_params=RayParams(num_actors=2, cpus_per_actor=1))

    bst.save_model("model.xgb")
    print("Final training error: {:.4f}".format(
        evals_result["train"]["error"][-1]))


The only differences compared to the non-distributed API are
the import statement (``xgboost_ray`` instead of ``xgboost``), using the
``RayDMatrix`` instead of the ``DMatrix``, and passing a :class:`RayParams <xgboost_ray.RayParams>` object.

The return object is a regular ``xgboost.Booster`` instance.


Simple prediction example
-------------------------
.. code-block:: python

    from xgboost_ray import RayDMatrix, RayParams, predict
    from sklearn.datasets import load_breast_cancer
    import xgboost as xgb

    data, labels = load_breast_cancer(return_X_y=True)

    dpred = RayDMatrix(data, labels)

    bst = xgb.Booster(model_file="model.xgb")
    pred_ray = predict(bst, dpred, ray_params=RayParams(num_actors=2))

    print(pred_ray)

In this example, the data will be split across two actors. The result array
will integrate this data in the correct order.

The RayParams object
========================
The ``RayParams`` object is used to configure various settings relating to
the distributed training.

.. autoclass:: xgboost_ray.RayParams

Multi GPU training
==================
Ray automatically detects GPUs on cluster nodes.
In order to start training on multiple GPUs, all you have to do is
to set the ``gpus_per_actor`` parameter of the ``RayParams`` object, as well
as the ``num_actors`` parameter for multiple GPUs:

.. code-block:: python

    ray_params = RayParams(
        num_actors=4,
        gpus_per_actor=1,
    )

This will train on four GPUs in parallel.

Note that it usually does not make sense to allocate more than one GPU per actor,
as XGBoost relies on distributed libraries such as Dask or Ray to utilize multi
GPU taining.

Setting the number of CPUs per actor
====================================
XGBoost natively utilizes multi threading to speed up computations. Thus if
your are training on CPUs only, there is likely no benefit in using more than
one actor per node. In that case, assuming you have a cluster of homogeneous nodes,
set the number of CPUs per actor to the number of CPUs available on each node,
and the number of actors to the number of nodes.

If you are using multi GPU training on a single node, divide the number of
available CPUs evenly across all actors. For instance, if you have 16 CPUs and
4 GPUs available, each actor should access 1 GPU and 4 CPUs.

If you are using a cluster of heterogeneous nodes (with different amounts of CPUs),
you might just want to use the `greatest common divisor <https://en.wikipedia.org/wiki/Greatest_common_divisor>`_
for the number of CPUs per actor. E.g. if you have a cluster of three nodes with
4, 8, and 12 CPUs, respectively, you'd start 6 actors with 4 CPUs each for maximum
CPU utilization.

Fault tolerance
===============
XGBoost-Ray supports two fault tolerance modes. In **non-elastic training**, whenever
a training actor dies (e.g. because the node goes down), the training job will stop,
XGBoost-Ray will wait for the actor (or its resources) to become available again
(this might be on a different node), and then continue training once all actors are back.

In **elastic-training**, whenever a training actor dies, the rest of the actors
continue training without the dead actor. If the actor comes back, it will be re-integrated
into training again.

Please note that in elastic-training this means that you will train on fewer data
for some time. The benefit is that you can continue training even if a node goes
away for the remainder of the training run, and don't have to wait until it is back up again.
In practice this usually leads to a very minor decrease in accuracy but a much shorter
training time compared to non-elastic training.

Both training modes can be configured using the respective :class:`RayParams <xgboost_ray.RayParams>`
parameters.

Hyperparameter optimization
===========================
XGBoost-Ray integrates well with `hyperparameter optimization framework Ray Tune <http://tune.io>`_.
Ray Tune uses Ray to start multiple distributed trials with different hyperparameter configurations.
If used with XGBoost-Ray, these trials will then start their own distributed training
jobs.

XGBoost-Ray automatically reports evaluation results back to Ray Tune. There's only
a few things you need to do:

1. Put your XGBoost-Ray training call into a function accepting parameter configurations
   (``train_model`` in the example below).
2. Create a :class:`RayParams <xgboost_ray.RayParams>` object (``ray_params``
   in the example below).
3. Define the parameter search space (``config`` dict in the example below).
4. Call ``tune.run()``:
    * The ``metric`` parameter should contain the metric you'd like to optimize.
      Usually this consists of the prefix passed to the ``evals`` argument of
      ``xgboost_ray.train()``, and an ``eval_metric`` passed in the
      XGBoost parameters (``train-error`` in the example below).
    * The ``mode`` should either be ``min`` or ``max``, depending on whether
      you'd like to minimize or maximize the metric
    * The ``resources_per_actor`` should be set using ``ray_params.get_tune_resources()``.
      This will make sure that each trial has the necessary resources available to
      start their distributed training jobs.

.. code-block:: python

    from xgboost_ray import RayDMatrix, RayParams, train
    from sklearn.datasets import load_breast_cancer

    num_actors = 4
    num_cpus_per_actor = 1

    ray_params = RayParams(
        num_actors=num_actors, cpus_per_actor=num_cpus_per_actor)

    def train_model(config):
        train_x, train_y = load_breast_cancer(return_X_y=True)
        train_set = RayDMatrix(train_x, train_y)

        evals_result = {}
        bst = train(
            params=config,
            dtrain=train_set,
            evals_result=evals_result,
            evals=[(train_set, "train")],
            verbose_eval=False,
            ray_params=ray_params)
        bst.save_model("model.xgb")

    from ray import tune

    # Specify the hyperparameter search space.
    config = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "eta": tune.loguniform(1e-4, 1e-1),
        "subsample": tune.uniform(0.5, 1.0),
        "max_depth": tune.randint(1, 9)
    }

    # Make sure to use the `get_tune_resources` method to set the `resources_per_trial`
    analysis = tune.run(
        train_model,
        config=config,
        metric="train-error",
        mode="min",
        num_samples=4,
        resources_per_trial=ray_params.get_tune_resources())
    print("Best hyperparameters", analysis.best_config)


Ray Tune supports various
`search algorithms and libraries (e.g. BayesOpt, Tree-Parzen estimators) <https://docs.ray.io/en/latest/tune/key-concepts.html#search-algorithms>`_,
`smart schedulers like successive halving <https://docs.ray.io/en/latest/tune/key-concepts.html#trial-schedulers>`_,
and other features. Please refer to the `Ray Tune documentation <http://tune.io>`_
for more information.

Additional resources
====================
* `XGBoost-Ray repository <https://github.com/ray-project/xgboost_ray>`_
* `XGBoost-Ray documentation <https://docs.ray.io/en/master/xgboost-ray.html>`_
* `Ray core documentation <https://docs.ray.io/en/master/index.html>`_
* `Ray Tune documentation <http://tune.io>`_
