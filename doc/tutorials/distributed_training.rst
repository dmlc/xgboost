#############################
Distributed Training Overview
#############################

This page provides a framework-agnostic map of how distributed XGBoost works.
Use it as a starting point before diving into integration-specific guides for
Dask, Spark, or Ray.

Why this guide exists
=====================

XGBoost already has interface-specific tutorials, but users building or
customizing distributed integrations often need one place that explains the
shared concepts first. This overview focuses on those common components.

Core components
===============

A distributed XGBoost workload usually includes four pieces:

1. **Orchestrator/runtime**

   The distributed framework coordinates process startup, scheduling, and
   retries (for example, Dask scheduler/workers, Spark driver/executors, or
   Ray driver/workers).

2. **Partitioned training data**

   Input data is split across workers. Each worker trains on a shard while the
   framework handles locality, movement, and task execution.

3. **Collective communication**

   Workers synchronize model updates through XGBoost's distributed communication
   layer. This is what keeps boosting rounds consistent across processes.

4. **Training/evaluation control**

   Parameters, evaluation metrics, and checkpointing are coordinated at the API
   layer provided by each integration.

Common workflow
===============

Most integrations follow the same high-level flow:

1. Start or connect to a distributed cluster.
2. Materialize data in the framework-native distributed format.
3. Construct the integration-specific XGBoost dataset/object.
4. Launch training with distributed parameters.
5. Run evaluation/prediction and persist model artifacts.

Choosing an integration
=======================

- Use :doc:`dask` for Python-native distributed data science workflows.
- Use :doc:`spark_estimator` (or XGBoost4J-Spark docs) for JVM/Spark pipelines.
- Use :doc:`ray` for Ray-native training and serving ecosystems.

Each guide contains API details and runnable examples. Start there once the
shared architecture above is clear.

Related documentation
=====================

- :doc:`dask`
- :doc:`spark_estimator`
- :doc:`ray`
- `Distributed XGBoost with XGBoost4J-Spark <https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html>`_
- `Distributed XGBoost with XGBoost4J-Spark-GPU <https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_gpu_tutorial.html>`_
