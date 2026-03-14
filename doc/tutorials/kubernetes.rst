###################################
Distributed XGBoost on Kubernetes
###################################

Distributed XGBoost training on `Kubernetes <https://kubernetes.io/>`_ is supported
via `Kubeflow Trainer <https://github.com/kubeflow/trainer>`_. Kubeflow Trainer provides
a built-in XGBoost runtime that manages the scheduling, distributed coordination, and
lifecycle of XGBoost training jobs on Kubernetes clusters.

This tutorial covers the end-to-end workflow: from setting up prerequisites, through
writing distributed training code, to launching and monitoring multi-node XGBoost jobs.

.. contents::
  :backlinks: none
  :local:

********
Overview
********

XGBoost supports distributed training through the **Collective** communication
protocol (historically known as Rabit). In a distributed setting, multiple worker
processes each operate on a shard of the data and synchronize histogram bin
statistics via AllReduce to agree on the best tree splits. Kubeflow Trainer's
XGBoost runtime automates the orchestration of this process on Kubernetes by:

- Deploying worker pods as a `JobSet <https://github.com/kubernetes-sigs/jobset>`_
- Automatically injecting the ``DMLC_*`` environment variables required by XGBoost's
  Collective communication layer
- Providing the rank-0 pod with the tracker address so user code can start a
  ``RabitTracker`` for worker coordination
- Supporting both CPU and GPU training workloads

Architecture
============

The distributed XGBoost training architecture on Kubernetes consists of the following
components:

1. **TrainJob**: A Kubernetes custom resource that declares the training job configuration
   (number of nodes, resources per node, training code).
2. **ClusterTrainingRuntime**: A cluster-scoped resource that defines the XGBoost runtime
   template (container image, ML policy, default settings). The built-in runtime is named
   ``xgboost-distributed``.
3. **Trainer Controller**: Resolves the ``TrainJob`` against the referenced runtime,
   enforces the XGBoost ML policy (injects environment variables), and creates the
   underlying ``JobSet``.
4. **Worker Pods**: Each pod runs the same training script. The user's training code
   on the rank-0 pod is responsible for starting a ``RabitTracker`` for coordination.

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │  User submits TrainJob (SDK or kubectl)                         │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Trainer Controller                                             │
   │  • Resolves ClusterTrainingRuntime (xgboost-distributed)        │
   │  • Enforces XGBoost MLPolicy (injects DMLC_* env vars)          │
   │  • Creates JobSet with worker pods                              │
   └──────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Kubernetes Cluster (Headless Service)                          │
   │                                                                 │
   │  ┌────────────────┐  ┌──────────┐  ┌──────────┐                 │
   │  │ Pod: node-0-0  │  │ node-0-1 │  │ node-0-2 │  ...            │
   │  │ TASK_ID=0      │  │ TASK_ID=1│  │ TASK_ID=2│                 │
   │  │ (Tracker)      │  │ (Worker) │  │ (Worker) │                 │
   │  └───────┬────────┘  └────┬─────┘  └────┬─────┘                 │
   │          │                │              │                      │
   │          └──── Collective Protocol ───────┘                     │
   └─────────────────────────────────────────────────────────────────┘

Environment Variables
=====================

The XGBoost runtime plugin automatically injects the following environment variables
into each worker pod. These are native to XGBoost's Collective protocol:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Variable
     - Description
     - Example Value
   * - ``DMLC_TRACKER_URI``
     - DNS address of the rank-0 pod running the tracker
     - ``myjob-node-0-0.myjob``
   * - ``DMLC_TRACKER_PORT``
     - Port for tracker communication
     - ``29500``
   * - ``DMLC_TASK_ID``
     - Worker rank (derived from pod completion index)
     - ``0``, ``1``, ``2``, ...
   * - ``DMLC_NUM_WORKER``
     - Total number of workers across all nodes
     - ``4``

These environment variables are **reserved** and cannot be manually set by the user in the
``TrainJob`` spec. The runtime plugin validates this and rejects any ``TrainJob`` that
attempts to override them.

Worker Count Calculation
========================

The total number of workers (``DMLC_NUM_WORKER``) is calculated as:

.. code-block:: text

   DMLC_NUM_WORKER = numNodes × workersPerNode

Where ``workersPerNode`` is determined by:

- **CPU training**: 1 worker per node. XGBoost does **not** spawn multiple worker
  processes for CPU training. Instead, a single worker process uses OpenMP to
  parallelize tree building across all available CPU cores on the node. This means
  if a pod has 8 CPU cores, 1 XGBoost worker will use all 8 cores for intra-process
  parallelism (histogram construction, split evaluation, etc.).

  The number of threads can be controlled with the ``nthread`` Booster parameter:

  .. code-block:: python

     # By default, XGBoost uses all available CPU cores.
     # Set nthread to limit the number of OpenMP threads per worker.
     params = {
         "objective": "binary:logistic",
         "nthread": 4,          # Use only 4 of the available cores
         "tree_method": "hist",
     }

  The ``nthread`` parameter in the DMatrix constructor controls parallelism during
  data loading, while ``nthread`` in the Booster parameters controls parallelism
  during training. If not set, both default to the maximum number of threads
  available on the machine.

  .. tip::

     When setting ``resourcesPerNode`` CPU requests in your ``TrainJob``, align the
     ``nthread`` parameter with the CPU requests to avoid over-subscription. For
     example, if you request ``cpu: "4"``, set ``"nthread": 4`` in your training
     parameters.

- **GPU training**: 1 worker per GPU. The GPU count is derived from the
  ``resourcesPerNode`` limits in the ``TrainJob`` or runtime template.  In
  distributed environments, use ``device="cuda"`` (not ``"cuda:<ordinal>"``);
  GPU ordinal selection is handled by the distributed framework, and specifying
  an ordinal will result in an error.

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 20

   * - Configuration
     - numNodes
     - workersPerNode
     - DMLC_NUM_WORKER
   * - 4 nodes, CPU-only
     - 4
     - 1
     - 4
   * - 2 nodes, 4 GPUs each
     - 2
     - 4
     - 8
   * - 1 node, 8 GPUs
     - 1
     - 8
     - 8

*************
Prerequisites
*************

Before running distributed XGBoost jobs on Kubernetes, ensure the following:

1. **Kubernetes Cluster**: A running Kubernetes cluster (v1.27+). You can use
   `kind <https://kind.sigs.k8s.io/>`_, `minikube <https://minikube.sigs.k8s.io/>`_,
   or a managed Kubernetes service (GKE, EKS, AKS).

2. **kubectl**: The Kubernetes CLI tool, configured to communicate with your cluster.
   See the `kubectl installation guide <https://kubernetes.io/docs/tasks/tools/>`_.

3. **Kubeflow Trainer**: Install Kubeflow Trainer and its dependencies (JobSet) on
   your cluster. Follow the
   `Kubeflow Trainer installation guide <https://www.kubeflow.org/docs/components/trainer/>`_:

   .. code-block:: bash

      # Install the Kubeflow Trainer control plane (includes JobSet).
      kubectl apply --server-side -k "github.com/kubeflow/trainer/manifests/overlays/standalone"

4. **Kubeflow Python SDK** (optional, for programmatic job submission):

   .. code-block:: bash

      pip install kubeflow

5. **GPU Support** (optional, for GPU training): Ensure the
   `NVIDIA GPU Operator <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html>`_
   or equivalent device plugin is installed on your cluster.

Verify the Installation
=======================

After installing Kubeflow Trainer, verify that the XGBoost runtime is available:

.. code-block:: bash

   kubectl get clustertrainingruntime

You should see the ``xgboost-distributed`` runtime listed:

.. code-block:: text

   NAME                   AGE
   xgboost-distributed    1m

************************************
XGBoost ClusterTrainingRuntime
************************************

The ``xgboost-distributed`` ``ClusterTrainingRuntime`` is deployed as part of the
Kubeflow Trainer installation. It defines the default XGBoost runtime template:

.. code-block:: yaml

   apiVersion: trainer.kubeflow.org/v1alpha1
   kind: ClusterTrainingRuntime
   metadata:
     name: xgboost-distributed
     labels:
       trainer.kubeflow.org/framework: xgboost
   spec:
     mlPolicy:
       numNodes: 1
       xgboost: {}
     template:
       spec:
         replicatedJobs:
           - name: node
             template:
               metadata:
                 labels:
                   trainer.kubeflow.org/trainjob-ancestor-step: trainer
               spec:
                 template:
                   spec:
                     containers:
                       - name: node
                         image: ghcr.io/kubeflow/trainer/xgboost-runtime:latest

Key points:

- ``mlPolicy.xgboost: {}`` activates the XGBoost runtime plugin, which handles
  injection of ``DMLC_*`` environment variables.
- ``numNodes`` defaults to ``1`` and can be overridden per ``TrainJob``.
- The container image ``ghcr.io/kubeflow/trainer/xgboost-runtime:latest`` is based on
  ``nvidia/cuda:12.4.0-runtime-ubuntu22.04`` and includes XGBoost 3.0.2 with CUDA 12
  support, NumPy, and scikit-learn.

***************************************
Example: Distributed XGBoost Training
***************************************

This section demonstrates two approaches for running distributed XGBoost training:
using the Python SDK (recommended for interactive use) and using ``kubectl`` with YAML
manifests.

Using the Python SDK
====================

The Kubeflow Python SDK provides a ``TrainerClient`` that simplifies submitting and
managing training jobs programmatically.

Step 1: Define the Training Function
-------------------------------------

Write the training function that will be serialized and executed on each worker node.
The ``DMLC_*`` environment variables are automatically injected by the runtime.

.. code-block:: python

   def xgboost_train_classification():
       """
       Distributed XGBoost training function using the Collective API.

       DMLC_* env vars are injected by the Kubeflow Trainer XGBoost plugin:
         - DMLC_TRACKER_URI:  DNS name of the rank-0 pod running the tracker
         - DMLC_TRACKER_PORT: Port for tracker communication (default: 29500)
         - DMLC_TASK_ID:      Worker rank (0, 1, 2, ...)
         - DMLC_NUM_WORKER:   Total number of workers
       """
       import os
       import xgboost as xgb
       from xgboost import collective as coll
       from xgboost.tracker import RabitTracker
       from sklearn.datasets import make_classification
       from sklearn.model_selection import train_test_split
       from sklearn.metrics import accuracy_score

       # Read injected environment variables.
       rank = int(os.environ["DMLC_TASK_ID"])
       world_size = int(os.environ["DMLC_NUM_WORKER"])
       tracker_uri = os.environ["DMLC_TRACKER_URI"]
       tracker_port = int(os.environ["DMLC_TRACKER_PORT"])

       # Rank 0 starts the Rabit tracker (required for coordination).
       tracker = None
       if rank == 0:
           tracker = RabitTracker(
               host_ip="0.0.0.0", n_workers=world_size, port=tracker_port
           )
           tracker.start()

       # All workers connect to the tracker via the Collective communicator.
       with coll.CommunicatorContext(
           dmlc_tracker_uri=tracker_uri,
           dmlc_tracker_port=tracker_port,
           dmlc_task_id=str(rank),
       ):
           # Generate synthetic classification data.
           # In practice, each worker would load its own data shard.
           X, y = make_classification(
               n_samples=10000, n_features=20, n_informative=10,
               n_classes=2, random_state=42 + rank,
           )
           X_train, X_valid, y_train, y_valid = train_test_split(
               X, y, test_size=0.2, random_state=42,
           )

           # NOTE: DMatrix construction MUST be inside the communicator context
           # because it involves cross-worker synchronization for quantization.
           #
           # Use QuantileDMatrix instead of DMatrix for the hist tree method
           # (the default). QuantileDMatrix quantizes data on-the-fly, avoiding
           # an intermediate dense copy and significantly reducing memory usage.
           dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
           # Validation QuantileDMatrix must reference the training matrix
           # so that the same quantile bins are reused.
           dvalid = xgb.QuantileDMatrix(X_valid, label=y_valid, ref=dtrain)

           # Training parameters.
           params = {
               "objective": "binary:logistic",
               "max_depth": 6,
               "eta": 0.1,
               "eval_metric": "logloss",
           }

           # Distributed training - workers synchronize histogram stats via collective ops.
           # early_stopping_rounds activates early stopping based on the validation metric.
           # verbose_eval=10 prints evaluation results every 10 rounds (rank 0 only).
           model = xgb.train(
               params, dtrain,
               num_boost_round=100,
               evals=[(dvalid, "validation")],
               early_stopping_rounds=10,
               verbose_eval=10,
           )

           # Note: early_stopping_rounds returns the *last* model, not the best.
           # Use bst.best_iteration to slice the model to the best round.
           if hasattr(model, "best_iteration"):
               model = model[: model.best_iteration + 1]

           # Evaluate on validation set.
           preds = model.predict(dvalid)
           predictions = [1 if p > 0.5 else 0 for p in preds]
           accuracy = accuracy_score(y_valid, predictions)

           # Only perform logging and model saving from rank 0
           # to avoid duplicate output and file write conflicts.
           if coll.get_rank() == 0:
               print(f"Validation Accuracy: {accuracy:.4f}")
               model.save_model("/workspace/xgboost_model.json")
               print("Model saved to /workspace/xgboost_model.json")

       # Wait for tracker to finish (rank 0 only).
       if tracker is not None:
           tracker.wait_for()

Step 2: Submit the Training Job
-------------------------------

Use the ``TrainerClient`` to submit the training function as a distributed job:

.. code-block:: python

   from kubeflow.trainer import CustomTrainer, TrainerClient

   client = TrainerClient()

   # Submit a distributed XGBoost training job on 3 nodes.
   job_name = client.train(
       trainer=CustomTrainer(
           func=xgboost_train_classification,
           num_nodes=3,
           resources_per_node={"cpu": 3},
       ),
       runtime="xgboost-distributed",
   )

   print(f"TrainJob '{job_name}' submitted")

For GPU training, include GPU resources:

.. code-block:: python

   job_name = client.train(
       trainer=CustomTrainer(
           func=xgboost_train_classification,
           num_nodes=2,
           resources_per_node={
               "cpu": 4,
               "gpu": 4,  # 4 GPUs per node → 8 total workers
           },
       ),
       runtime="xgboost-distributed",
   )

.. note::

   For GPU training, add ``"device": "cuda"`` to the XGBoost ``params`` dictionary
   in your training function.

Step 3: Monitor the Training Job
---------------------------------

Check the job status and view logs:

.. code-block:: python

   # Wait for the job to start running.
   client.wait_for_job_status(name=job_name, status={"Running"})

   # Check the steps (one per worker node).
   for step in client.get_job(name=job_name).steps:
       print(f"Step: {step.name}, Status: {step.status}")

   # Stream logs from each worker node.
   num_nodes = 3
   for i in range(num_nodes):
       logs = client.get_job_logs(name=job_name, follow=True, step=f"node-{i}")
       print(f"\n=== Node {i} ===")
       print("\n".join(logs))

Step 4: Clean Up
----------------

Delete the training job when it is finished:

.. code-block:: python

   client.delete_job(job_name)

Using kubectl with YAML
========================

You can also create ``TrainJob`` resources directly using ``kubectl``.

CPU Training Example
---------------------

The following YAML creates a distributed XGBoost training job with 4 worker nodes:

.. code-block:: yaml

   apiVersion: trainer.kubeflow.org/v1alpha1
   kind: TrainJob
   metadata:
     name: xgboost-cpu-example
   spec:
     runtimeRef:
       name: xgboost-distributed
     trainer:
       image: ghcr.io/kubeflow/trainer/xgboost-runtime:latest
       command:
         - python
         - train.py
       numNodes: 4
       resourcesPerNode:
         requests:
           cpu: "4"
           memory: "8Gi"

Apply the manifest:

.. code-block:: bash

   kubectl apply -f xgboost-cpu-trainjob.yaml

GPU Training Example
---------------------

For multi-node GPU training, specify GPU resources via ``resourcesPerNode``:

.. code-block:: yaml

   apiVersion: trainer.kubeflow.org/v1alpha1
   kind: TrainJob
   metadata:
     name: xgboost-gpu-example
   spec:
     runtimeRef:
       name: xgboost-distributed
     trainer:
       image: ghcr.io/kubeflow/trainer/xgboost-runtime:latest
       command:
         - python
         - train.py
       numNodes: 2
       resourcesPerNode:
         limits:
           nvidia.com/gpu: "4"
         requests:
           cpu: "4"
           memory: "16Gi"

With this configuration, the runtime calculates ``DMLC_NUM_WORKER = 2 nodes × 4 GPUs = 8``.
Each GPU runs one XGBoost worker process.

Monitoring with kubectl
------------------------

.. code-block:: bash

   # Check TrainJob status.
   kubectl get trainjob xgboost-cpu-example

   # View logs from a specific worker pod.
   kubectl logs xgboost-cpu-example-node-0-0

   # Delete the TrainJob.
   kubectl delete trainjob xgboost-cpu-example

*************
How It Works
*************

This section provides additional implementation details for users who want to
understand the runtime plugin internals.

XGBoost Runtime Plugin
======================

The XGBoost runtime is implemented as a Go plugin in the Kubeflow Trainer controller
(see ``pkg/runtime/framework/plugins/xgboost/`` in the Trainer repository). It
implements two interfaces:

- ``EnforceMLPolicyPlugin``: Injects the ``DMLC_*`` environment variables (described
  in `Environment Variables`_) and exposes container port ``29500``.
- ``CustomValidationPlugin``: Rejects any ``TrainJob`` that manually sets reserved
  ``DMLC_*`` environment variables.

Tracker Discovery
=================

Workers discover the ``RabitTracker`` on rank-0 via a Kubernetes headless service.
The ``DMLC_TRACKER_URI`` is constructed as:

.. code-block:: text

   <trainjob-name>-node-0-0.<trainjob-name>

For example, a ``TrainJob`` named ``myjob`` with 4 nodes creates pods:

.. code-block:: text

   myjob-node-0-0   DMLC_TASK_ID=0   (Tracker + Worker)
   myjob-node-0-1   DMLC_TASK_ID=1   (Worker)
   myjob-node-0-2   DMLC_TASK_ID=2   (Worker)
   myjob-node-0-3   DMLC_TASK_ID=3   (Worker)

.. note::

   Starting the tracker is the **user's responsibility**. The runtime injects the
   environment variables, but the training code on rank-0 must call
   ``RabitTracker(...).start()`` before other workers can connect.

***************
Best Practices
***************

This section covers practical tips for getting the most out of distributed XGBoost
on Kubernetes.

Use QuantileDMatrix for Memory Efficiency
=========================================

The default tree method is ``hist`` (``tree_method="auto"`` resolves to ``hist``).
When using ``hist``, prefer :py:class:`xgboost.QuantileDMatrix` over
:py:class:`xgboost.DMatrix`. ``QuantileDMatrix`` generates quantilized data directly
from input, skipping the intermediate dense representation and significantly
reducing memory consumption:

.. code-block:: python

   # Standard DMatrix — loads data then quantizes (higher peak memory)
   dtrain = xgb.DMatrix(X_train, label=y_train)

   # QuantileDMatrix — quantizes on-the-fly (lower peak memory)
   dtrain = xgb.QuantileDMatrix(X_train, label=y_train)

When constructing a validation ``QuantileDMatrix``, always pass the training matrix
as ``ref`` so XGBoost reuses the same quantile bins. Omitting ``ref`` for validation
data may lead to inconsistent quantization and degraded model quality:

.. code-block:: python

   dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
   dvalid = xgb.QuantileDMatrix(X_valid, label=y_valid, ref=dtrain)  # correct

.. note::

   ``QuantileDMatrix`` was added in XGBoost 1.7.0. No explicit ``tree_method``
   parameter is needed — the default ``auto`` already uses ``hist``.

Early Stopping
==============

Early stopping is activated by passing ``early_stopping_rounds`` to
:py:func:`xgboost.train`. It requires at least one validation set in ``evals``.
Training stops if the validation metric does not improve for the specified number
of consecutive rounds:

.. code-block:: python

   model = xgb.train(
       params, dtrain,
       num_boost_round=500,
       evals=[(dvalid, "validation")],
       early_stopping_rounds=10,
   )

Early stopping works correctly in distributed mode — evaluation metrics are already
synchronized across workers via the collective protocol.

**Important**: ``xgb.train`` with ``early_stopping_rounds`` returns the **last**
model, not the best one. To get the best model, use model slicing:

.. code-block:: python

   # After training, slice to keep only rounds up to the best iteration.
   if hasattr(model, "best_iteration"):
       model = model[: model.best_iteration + 1]

Alternatively, use the :py:class:`xgboost.callback.EarlyStopping` callback directly
with ``save_best=True`` to automatically keep only the best model:

.. code-block:: python

   from xgboost.callback import EarlyStopping

   model = xgb.train(
       params, dtrain,
       num_boost_round=500,
       evals=[(dvalid, "validation")],
       callbacks=[EarlyStopping(rounds=10, save_best=True)],
   )
   # model now contains only the rounds up to the best iteration

When multiple evaluation datasets are provided in ``evals``, the **last** entry
is used for early stopping. When multiple ``eval_metric`` values are specified,
the **last** metric is used.

Logging in Distributed Mode
===========================

In distributed training, ``print()`` executes on every worker, producing duplicate
log lines. To log from a single worker, guard with a rank check:

.. code-block:: python

   from xgboost import collective as coll

   with coll.CommunicatorContext(...):
       # Print only from rank 0.
       if coll.get_rank() == 0:
           print(f"Training complete, best score: {model.best_score}")

:py:func:`xgboost.collective.communicator_print` is an alternative that routes
messages through the tracker rather than stdout. Note that it does **not** filter
by rank — any worker that calls it will have its message printed by the tracker.
It is primarily used internally (e.g., by ``verbose_eval``, which adds its own
rank-0 guard via :py:class:`xgboost.callback.EvaluationMonitor`).

Setting verbose_eval for Production
===================================

In distributed Kubernetes jobs, set ``verbose_eval`` to an integer rather than
``True`` to reduce log volume:

.. code-block:: python

   model = xgb.train(
       params, dtrain,
       num_boost_round=500,
       evals=[(dvalid, "validation")],
       verbose_eval=50,  # print every 50 rounds instead of every round
   )

Checkpointing
=============

XGBoost provides a :py:class:`xgboost.callback.TrainingCheckPoint` callback that
periodically saves model snapshots during training. The callback automatically
saves only from rank 0 to avoid multiple workers writing to the same path:

.. code-block:: python

   from xgboost.callback import TrainingCheckPoint

   model = xgb.train(
       params, dtrain,
       num_boost_round=500,
       evals=[(dvalid, "validation")],
       callbacks=[
           TrainingCheckPoint(
               directory="/workspace/checkpoints",
               name="xgb_model",
               interval=50,  # save every 50 rounds
           ),
       ],
   )

.. warning::

   XGBoost does not handle distributed file systems. The ``directory`` path must be
   writable from the rank-0 pod — for example, a Kubernetes
   `PersistentVolumeClaim <https://kubernetes.io/docs/concepts/storage/persistent-volumes/>`_
   mounted into the pod.

To resume training from a checkpoint, pass the saved model file via ``xgb_model``:

.. code-block:: python

   model = xgb.train(
       params, dtrain,
       num_boost_round=500,
       xgb_model="/workspace/checkpoints/xgb_model_200.ubj",  # resume from round 200
       evals=[(dvalid, "validation")],
   )

Data Partitioning
=================

By default, each worker in a distributed XGBoost job holds a different subset of
**rows** (horizontal partitioning). This is controlled by the ``data_split_mode``
parameter (default: ``DataSplitMode.ROW``). In this mode, each worker loads its
own shard of the data:

.. code-block:: python

   with coll.CommunicatorContext(...):
       # Each worker loads a different data shard based on its rank.
       rank = coll.get_rank()
       X_shard, y_shard = load_data_shard(rank)
       dtrain = xgb.QuantileDMatrix(X_shard, label=y_shard)

Column-wise splitting (``DataSplitMode.COL``) is also supported, where each worker
holds a different subset of features. This is typically used for vertical federated
learning scenarios and is not the common distributed training pattern.

Rank-Specific Logic
===================

Use :py:func:`xgboost.collective.get_rank` and
:py:func:`xgboost.collective.get_world_size` for rank-specific operations inside
the communicator context:

.. code-block:: python

   with coll.CommunicatorContext(...):
       if coll.get_rank() == 0:
           model.save_model("/workspace/model.json")
           # Broadcast results to all workers if needed
           results = coll.broadcast(results, root=0)

:py:func:`xgboost.collective.broadcast` can broadcast any picklable Python object
from one worker to all others. This is useful for sharing preprocessed metadata
(e.g., label encoders, feature name lists) computed on rank 0.

********************************
Common Issues and Edge Cases
********************************

Reserved Environment Variables
==============================

The runtime plugin rejects any ``TrainJob`` that manually sets the reserved ``DMLC_*``
environment variables (``DMLC_TRACKER_URI``, ``DMLC_TRACKER_PORT``, ``DMLC_TASK_ID``,
``DMLC_NUM_WORKER``). If you set any of these in ``spec.trainer.env``, the webhook
will return a ``Forbidden`` error:

.. code-block:: text

   spec.trainer.env[0]: Forbidden: DMLC_TRACKER_URI is reserved for the XGBoost runtime

Remove the reserved variables from your ``TrainJob`` spec and let the runtime inject
them automatically.

No Environment Injection When Trainer Is Nil
============================================

If the ``TrainJob`` does not include a ``spec.trainer`` section, the XGBoost plugin
skips environment variable injection entirely. The ``DMLC_*`` variables are only
injected when ``spec.trainer`` is present and the runtime can locate the ``node``
container in the pod template. Ensure your ``TrainJob`` includes the ``trainer``
field.

Resource Precedence: TrainJob Overrides Runtime
===============================================

When GPU resources are specified in both the ``ClusterTrainingRuntime`` template and
the ``TrainJob.spec.trainer.resourcesPerNode``, the **TrainJob value takes precedence**.
This affects the ``workersPerNode`` calculation:

.. code-block:: text

   Runtime template: nvidia.com/gpu: 1  →  workersPerNode = 1
   TrainJob override: nvidia.com/gpu: 3  →  workersPerNode = 3  (this wins)

If neither specifies GPU resources, ``workersPerNode`` defaults to ``1`` (CPU mode).

GPU Device Ordinal in Distributed Mode
======================================

In distributed training, do **not** use ``device="cuda:0"`` or any specific GPU ordinal
in your XGBoost parameters. GPU device assignment is handled by the Kubernetes device
plugin and the distributed framework. Use ``device="cuda"`` instead:

.. code-block:: python

   # Correct
   params = {"device": "cuda", "tree_method": "hist"}

   # Wrong — will raise an error in distributed mode
   params = {"device": "cuda:0", "tree_method": "hist"}

Data Matrices Must Be Inside CommunicatorContext
=================================================

Constructing ``xgb.DMatrix`` or ``xgb.QuantileDMatrix`` outside the
``CommunicatorContext`` may appear to work with dense data, but the behavior is
undefined. The constructor performs cross-worker synchronization for data shape
validation and quantile sketching (needed by ``tree_method="hist"``). Always
construct data matrices inside the context:

.. code-block:: python

   # Wrong — data matrix outside context
   dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
   with coll.CommunicatorContext(...):
       model = xgb.train(params, dtrain, ...)  # Undefined behavior

   # Correct — data matrix inside context
   with coll.CommunicatorContext(...):
       dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
       model = xgb.train(params, dtrain, ...)

Single-Node Defaults
====================

If ``numNodes`` is not specified in the ``TrainJob``, the runtime uses the default
from the ``ClusterTrainingRuntime`` (``1`` for the ``xgboost-distributed`` runtime).
A single-node job still goes through the full runtime pipeline — the ``RabitTracker``
is started on rank-0 (which is the only pod), and ``DMLC_NUM_WORKER`` is set to ``1``.
This is useful for testing your training function locally before scaling up.

CPU Over-Subscription
=====================

By default, XGBoost uses all available CPU cores via OpenMP. In a Kubernetes pod,
"available cores" is determined by cgroup limits set by the container runtime.
If your pod specifies only CPU **requests** (no **limits**), the cgroup may not
cap CPU usage, and XGBoost may attempt to use all cores on the node, causing
contention with other pods.

To avoid this, either:

- Set ``nthread`` in your XGBoost parameters to match your CPU request
- Set CPU ``limits`` (not just ``requests``) in ``resourcesPerNode`` so the container
  runtime enforces a cgroup ceiling

.. code-block:: yaml

   # Setting both requests and limits ensures XGBoost sees the correct core count
   resourcesPerNode:
     requests:
       cpu: "4"
     limits:
       cpu: "4"

*******
Support
*******

- For issues related to the Kubeflow Trainer XGBoost runtime, open an issue on the
  `Kubeflow Trainer repository <https://github.com/kubeflow/trainer/issues>`_.
- For XGBoost-specific questions, see the
  `XGBoost documentation <https://xgboost.readthedocs.io/>`_.
- The complete example notebook is available in the
  `Kubeflow Trainer examples <https://github.com/kubeflow/trainer/tree/master/examples/xgboost/distributed-training>`_.
