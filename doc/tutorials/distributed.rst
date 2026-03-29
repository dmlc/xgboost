##################################
Distributed Training: An Overview
##################################

This page provides a unified entry point for distributed XGBoost training across
different backends. It explains shared concepts, backend-specific trade-offs,
and links to detailed tutorials.

.. note::

  This overview focuses on distributed training workflows. For single-node usage,
  see package-specific getting started guides.

**Contents**

.. contents::
  :backlinks: none
  :local:

***************
When to use what
***************

Use the table below as a quick decision guide.

.. list-table::
   :header-rows: 1
   :widths: 20 24 24 32

   * - Backend
     - Best for
     - Typical environment
     - Notes
   * - Dask
     - Python-native distributed training and data pipelines
     - Dask cluster (local or multi-node)
     - Great if your data/ETL is already in Dask.
   * - PySpark
     - Spark ML pipeline integration
     - Spark cluster (standalone/YARN/K8s)
     - Best fit when model training lives inside Spark jobs.
   * - Kubernetes (Kubeflow Trainer)
     - Platform-managed distributed jobs
     - Kubernetes-native MLOps
     - Strong for job orchestration, resource isolation, and lifecycle automation.

See also:
- Dask tutorial: :doc:`dask`
- PySpark tutorial: :doc:`spark_estimator`
- Kubernetes tutorial: :doc:`kubernetes`

*********************
Shared mental model
*********************

Regardless of backend, distributed XGBoost usually follows the same pattern:

1. Split data across workers.
2. Each worker computes local histogram / gradient statistics.
3. Workers synchronize statistics via collective communication.
4. All workers agree on tree split decisions for each boosting iteration.

Key terms:

- **Worker**: A training process participating in distributed coordination.
- **World size**: Total number of participating workers.
- **Collective communication**: Synchronization primitive used by distributed training.
- **Tracker / rendezvous**: Component that helps workers discover peers and coordinate startup.

*******************************
Backend-specific architecture
*******************************

Dask
====

- Entry points: :mod:`xgboost.dask`
- Typical APIs:
  - ``xgboost.dask.train``
  - ``xgboost.dask.predict``
  - ``xgboost.dask.inplace_predict``
  - ``DaskXGBRegressor`` / ``DaskXGBClassifier``

See :doc:`dask` for cluster setup and API examples.

PySpark
=======

- Entry points: :mod:`xgboost.spark`
- Typical estimators:
  - ``SparkXGBRegressor``
  - ``SparkXGBClassifier``
  - ``SparkXGBRanker``
- Integrates with Spark ML pipelines and DataFrame-based workflows.

See :doc:`spark_estimator` for estimator API and GPU usage notes.

Kubernetes (Kubeflow Trainer)
=============================

- Uses Kubernetes-native job orchestration for distributed XGBoost.
- Runtime injects distributed environment variables for worker coordination.
- Suitable for platform-managed, reproducible training jobs.

See :doc:`kubernetes` for end-to-end setup and runtime details.

**************************
Parameter alignment cheat sheet
**************************

The same intent may be configured differently depending on backend.
Use this table as a quick map.

.. list-table::
   :header-rows: 1
   :widths: 24 24 24 28

   * - Intent
     - Dask
     - PySpark
     - Kubernetes
   * - Number of workers
     - Cluster workers / partitions
     - ``num_workers`` on Spark estimator
     - Derived from job spec / runtime config
   * - Device selection
     - Booster param ``device`` (e.g. ``cpu`` / ``cuda``)
     - Estimator param ``device``
     - Runtime + pod resources; use distributed-safe ``device`` setting
   * - Thread parallelism
     - Booster param ``nthread`` per worker
     - Bound by Spark task CPU allocation
     - Bound by container CPU and training params
   * - Data interface
     - Dask Array/DataFrame
     - Spark DataFrame / feature columns
     - User training script + mounted/in-cluster data source

.. note::

  In distributed environments, avoid hard-coding per-worker GPU ordinals unless
  explicitly required by the backend runtime.

****************
Common pitfalls
****************

1. **Resource mismatch**
   - Worker count, CPU/GPU allocation, and thread settings are not aligned.
2. **Data partitioning imbalance**
   - Very small/uneven partitions can hurt throughput.
3. **Backend-specific assumptions**
   - API semantics differ across Dask and PySpark wrappers.
4. **Environment drift**
   - Inconsistent package/runtime versions across workers causes failures.

************************
A practical start sequence
************************

If you are new to distributed XGBoost:

1. Start with one backend that matches your current stack.
2. Run a minimal end-to-end example from the backend tutorial.
3. Validate scaling from 1 worker to N workers.
4. Tune worker count and thread/device settings.
5. Add evaluation and monitoring before large-scale runs.

********************
Where to go next
********************

- Dask distributed tutorial: :doc:`dask`
- PySpark estimator tutorial: :doc:`spark_estimator`
- Kubernetes distributed tutorial: :doc:`kubernetes`

If you are contributing docs, see:
- Documentation contributor guide: :doc:`../contrib/docs`
