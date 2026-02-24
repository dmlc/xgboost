##############################
Federated Learning with XGBoost
##############################

Federated learning enables model training across multiple parties without sharing raw
data. This tutorial starts with basic (non-homomorphic) federated training for both
horizontal and vertical settings, then explains security risks, and finally covers
secure solutions and plugin internals.

**Contents**

.. contents::
  :backlinks: none
  :local:

********
Overview
********

XGBoost supports two federated learning paradigms:

**Horizontal Federated Learning**
  Data is partitioned by rows. Each party has the same feature schema but different
  samples.

**Vertical Federated Learning**
  Data is partitioned by columns. Parties share sample identity (for example, aligned
  IDs after PSI) but hold different feature subsets.

A typical deployment has one federated server and multiple workers:

.. code-block:: none

                    +------------------+
                    | Federated Server |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                |                |
      +-----v-----+    +-----v-----+    +-----v-----+
      | Worker 0  |    | Worker 1  |    | Worker 2  |
      +-----------+    +-----------+    +-----------+

From XGBoost's perspective, the federated server is purely an orchestration component —
it coordinates collective communication operations (allgather, broadcast, etc.) between
workers but does not perform any training computation itself. All tree building, gradient
computation, and histogram construction happen on the workers.

In **horizontal FL** (row-split), the training loop proceeds as follows:

1. **Sketch synchronization** (once, before training): each worker builds a local weighted
   quantile sketch from its rows. All local sketches are gathered via
   ``collective::AllgatherV`` and merged into a single global sketch. The resulting
   histogram bin boundaries (cut points) are identical on every worker for the whole training process.
2. **Local histogram construction** (each boosting round): each worker computes G/H
   (gradient/Hessian) values for its local rows and accumulates them into per-node
   histograms using the shared bin boundaries.
3. **Histogram allreduce** (each boosting round): local histograms are summed across
   workers via ``collective::Allreduce(Op::kSum)``. After this step every worker holds the
   same aggregated histogram.
4. **Split finding**: every worker evaluates splits on the aggregated histogram
   independently — because the histograms and bins are identical, all workers arrive at the
   same best split.
5. **Row partition update**: each worker partitions its local rows according to the agreed
   split and proceeds to the next tree node.

In **vertical FL** (column-split), each worker holds all rows but only a subset of
features:

1. **Sketch construction** (once, before training): each worker builds sketches only for
   its own features. No cross-worker sketch merge is needed because features are disjoint.
2. **Gradient broadcast** (each boosting round): the active party (rank 0) computes
   gradients from the labels it owns, then sends them to all passive parties via
   ``collective::Broadcast`` so every worker can build histograms for its features.
3. **Local histogram construction** (each boosting round): each worker builds histograms
   for its own features over all rows. Because every worker has all rows, these histograms
   are already complete — no histogram allreduce is needed.
4. **Split finding + allgather**: each worker evaluates splits on its own features using
   the plaintext histograms and finds its local best candidate. ``collective::Allgather``
   then exchanges all candidates so that every worker receives every other worker's best
   candidate; each worker locally reduces the full list to select the globally best split.
5. **Tree structure update**: all workers apply the chosen global best split to update their row
   partitions.

Even though raw data is never shared, intermediate values such as gradients and histograms
exchanged between workers via the server can potentially leak information about the
underlying data. To address this, XGBoost supports homomorphic encryption (HE) through a
plugin mechanism.

Homomorphic encryption is a class of cryptographic schemes that allow
computation on encrypted data without decrypting it first. For vertical FL, Paillier
encryption protects gradient and histogram exchanges; for horizontal FL, CKKS encryption
secures histogram aggregation. Security concerns, threat models, and the HE solutions
are discussed in detail in later sections.

***********
Quick Start
***********

This section covers baseline federated training without homomorphic encryption. It is
useful for validating data preparation, orchestration, and convergence before adding
secure aggregation.

Common Requirements
===================

* XGBoost built with federated communication support (``dmlc_communicator='federated'``)
* Reachable server address and matching ``federated_world_size`` / ``federated_rank``
* DMatrix's ``data_split_mode``: set to ``0`` (default) for horizontal FL (row-split) or ``1`` for
  vertical FL (column-split)
* Consistent feature schema across parties for horizontal FL, or aligned sample IDs
  (e.g. via PSI) for vertical FL
* For vertical FL, rank 0 is the label owner (active party); other ranks load features
  only

Preparing Sample Data
=====================

The following snippet generates a synthetic binary classification dataset and writes it
to disk in the layout expected by the horizontal and vertical examples below. Run this
once before starting the server and workers.

.. code-block:: python

    import os
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    WORLD_SIZE = 3
    BASE = "/tmp/xgboost/federated"
    SEED = 42

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_classes=2, random_state=SEED,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
    )

    # --- Horizontal split (row-split) ---
    # Each site gets a disjoint subset of training rows with all columns.
    # The validation set is kept identical across all sites for easy comparison.
    # Column 0 is the label.
    train_idx = np.array_split(np.arange(len(X_train)), WORLD_SIZE)
    df_valid = pd.DataFrame(np.column_stack([y_valid, X_valid]))

    for i in range(WORLD_SIZE):
        site_dir = os.path.join(BASE, "horizontal", f"site-{i + 1}")
        os.makedirs(site_dir, exist_ok=True)

        df_train = pd.DataFrame(
            np.column_stack([y_train[train_idx[i]], X_train[train_idx[i]]])
        )
        df_train.to_csv(os.path.join(site_dir, "train.csv"), index=False, header=False)
        df_valid.to_csv(os.path.join(site_dir, "valid.csv"), index=False, header=False)

    # --- Vertical split (column-split) ---
    # Each site gets all rows but a disjoint subset of feature columns.
    # Site-1 (rank 0) also gets the label as column 0.
    feature_splits = np.array_split(np.arange(X_train.shape[1]), WORLD_SIZE)

    for i in range(WORLD_SIZE):
        site_dir = os.path.join(BASE, "vertical", f"site-{i + 1}")
        os.makedirs(site_dir, exist_ok=True)

        cols_train = X_train[:, feature_splits[i]]
        cols_valid = X_valid[:, feature_splits[i]]

        if i == 0:
            cols_train = np.column_stack([y_train, cols_train])
            cols_valid = np.column_stack([y_valid, cols_valid])

        pd.DataFrame(cols_train).to_csv(
            os.path.join(site_dir, "train.csv"), index=False, header=False,
        )
        pd.DataFrame(cols_valid).to_csv(
            os.path.join(site_dir, "valid.csv"), index=False, header=False,
        )

    # --- Centralized (non-federated) baseline ---
    # Full training and validation sets for comparison.
    central_dir = os.path.join(BASE, "centralized")
    os.makedirs(central_dir, exist_ok=True)
    pd.DataFrame(np.column_stack([y_train, X_train])).to_csv(
        os.path.join(central_dir, "train.csv"), index=False, header=False,
    )
    df_valid.to_csv(
        os.path.join(central_dir, "valid.csv"), index=False, header=False,
    )

    print(f"Data written to {BASE}/horizontal/, {BASE}/vertical/, and {BASE}/centralized/")

Horizontal FL (No Homomorphic Encryption)
=========================================

In horizontal FL, each worker trains on local rows while participating in collective
communication. Every party owns the label column for its samples (``data_split_mode=0``
is the default for row-split).

**Server:**

.. code-block:: python

    import xgboost.federated

    xgboost.federated.run_federated_server(
        n_workers=3,
        port=9091,
    )

**Worker (run once per party):**

.. code-block:: python

    import xgboost as xgb

    rank = 0  # 0, 1, ..., world_size - 1

    communicator_env = {
        "dmlc_communicator": "federated",
        "federated_server_address": "localhost:9091",
        "federated_world_size": 3,
        "federated_rank": rank,
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        train_path = f"/tmp/xgboost/federated/horizontal/site-{rank + 1}/train.csv"
        valid_path = f"/tmp/xgboost/federated/horizontal/site-{rank + 1}/valid.csv"

        # In horizontal FL every party has labels.
        dtrain = xgb.DMatrix(train_path + "?format=csv&label_column=0")
        dvalid = xgb.DMatrix(valid_path + "?format=csv&label_column=0")

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": 3,
            "eta": 0.1,
            "nthread": 1,
        }

        bst = xgb.train(
            params, dtrain, num_boost_round=3,
            evals=[(dvalid, "eval"), (dtrain, "train")],
        )

        # In horizontal FL all ranks produce the same model;
        # saving on rank 0 is sufficient.
        if xgb.collective.get_rank() == 0:
            bst.save_model("/tmp/xgboost/federated/horizontal_fed_model.json")

Vertical FL (No Homomorphic Encryption)
=======================================

In vertical FL, parties hold different feature columns for the same aligned samples.
Baseline vertical training can run without homomorphic encryption, but intermediate
statistics are not cryptographically protected.

.. note::
   Before vertical training, sample alignment (for example via PSI) must be completed
   outside this API so all parties operate on the same sample ordering.

In vertical FL, **rank 0 is the label owner (active party)** and all other ranks are
feature-only owners (passive parties). When loading data, rank 0 includes the label
column while other ranks load only their feature columns. The ``data_split_mode=1``
flag tells XGBoost the data is column-split.

**Server:**

.. code-block:: python

    import xgboost.federated

    xgboost.federated.run_federated_server(
        n_workers=3,
        port=9091,
    )

**Worker (run once per party):**

.. code-block:: python

    import xgboost as xgb

    rank = 1  # 0 for label owner, 1..N-1 for feature owners

    communicator_env = {
        "dmlc_communicator": "federated",
        "federated_server_address": "localhost:9091",
        "federated_world_size": 3,
        "federated_rank": rank,
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        train_path = f"/tmp/xgboost/federated/vertical/site-{rank + 1}/train.csv"
        valid_path = f"/tmp/xgboost/federated/vertical/site-{rank + 1}/valid.csv"

        # Rank 0 (active party) owns the label column;
        # other ranks load feature columns only.
        if rank == 0:
            label = "&label_column=0"
        else:
            label = ""

        dtrain = xgb.DMatrix(
            train_path + f"?format=csv{label}", data_split_mode=1
        )
        dvalid = xgb.DMatrix(
            valid_path + f"?format=csv{label}", data_split_mode=1
        )

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": 3,
            "eta": 0.1,
            "nthread": 1,
        }

        bst = xgb.train(
            params, dtrain, num_boost_round=3,
            evals=[(dvalid, "eval"), (dtrain, "train")],
        )

        # Each rank saves its own model slice (split values
        # for features it does not own are stored as NaN).
        bst.save_model(f"/tmp/xgboost/federated/vertical_model.{rank}.json")

Comparing Against Centralized Training
=======================================

To verify that federated training produces equivalent results, train a centralized
(non-federated) baseline on the full dataset and compare validation metrics. The data
prep above already writes the unsplit data to ``/tmp/xgboost/federated/centralized/``.

.. code-block:: python

    import xgboost as xgb

    dtrain = xgb.DMatrix(
        "/tmp/xgboost/federated/centralized/train.csv?format=csv&label_column=0"
    )
    dvalid = xgb.DMatrix(
        "/tmp/xgboost/federated/centralized/valid.csv?format=csv&label_column=0"
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.1,
        "nthread": 1,
    }

    bst = xgb.train(
        params, dtrain, num_boost_round=3,
        evals=[(dvalid, "eval"), (dtrain, "train")],
    )

    bst.save_model("/tmp/xgboost/federated/centralized_model.json")

Vertical federated training should produce a model that is exactly identical to the
centralized baseline, because every party holds all samples and the histogram
aggregation across column-split parties yields the same splits as centralized histogram
construction.

Horizontal federated training also synchronizes histogram bin boundaries before
training begins: each worker builds a local weighted quantile sketch, then all sketches
are gathered and merged into a single global sketch via collective allreduce. The
resulting cut points are identical on every worker. During each boosting round, local
G/H histograms are summed across workers, so all parties see the same aggregated
histograms and make the same split decisions. In practice the horizontal federated
model should be very close to the centralized baseline. The only potential source of
minor numerical difference is the quantile sketch approximation — merging N local
sketches is not always bit-identical to building one sketch on all data at once.

To confirm, compare the saved model JSON files:

.. code-block:: python

    import json

    def load_model_json(path):
        with open(path) as f:
            return json.load(f)

    centralized = load_model_json("/tmp/xgboost/federated/centralized_model.json")

    # Vertical: rank 0 holds the complete tree structure (other ranks
    # store NaN for features they do not own, so rank 0 is the one to compare).
    vertical = load_model_json("/tmp/xgboost/federated/vertical_model.0.json")

    horizontal = load_model_json("/tmp/xgboost/federated/horizontal_fed_model.json")

    # The vertical model from rank 0 should be identical to centralized.
    print("vertical == centralized:", centralized == vertical)

    # The horizontal model should be very close; tree structures and
    # split values may differ only due to sketch approximation.
    print("horizontal == centralized:", centralized == horizontal)

*********************************************************
Secure Federated Learning with Homomorphic Encryption
*********************************************************

The non-secure pipelines above exchange gradients, histograms, and split candidates in
plaintext. Even though raw data is never shared, these intermediate values can leak
information about the underlying data. XGBoost addresses this through homomorphic
encryption (HE) via a plugin mechanism.

Threat Model and Solutions
=================================

This tutorial assumes an **honest-but-curious** setting: parties follow the protocol but
may inspect received data to infer private information.

The table below lists the key risks in each FL mode and how HE addresses them. All
branches in the code are gated by ``collective::IsEncrypted()``.

**Horizontal FL risks**

* Per-party histograms summed via ``collective::Allreduce`` are visible in plaintext at
  each worker and at the aggregation endpoint. *Solution*: each worker encrypts its
  histogram (``BuildEncryptedHistHori``) before the allgather; decryption
  (``SyncEncryptedHistHori``) happens locally after aggregation.

**Vertical FL risks**

* Plaintext gradients broadcast from the active party allow passive parties to
  reconstruct label information. *Solution*: gradients are encrypted before
  ``collective::Broadcast``; passive parties only see ciphertext.
* Plaintext histograms gathered via ``collective::Allgather`` expose per-feature
  split-gain information to all parties. *Solution*: passive parties build encrypted
  histograms (``BuildEncryptedHistVert``); only the active party decrypts
  (``SyncEncryptedHistVert``) and evaluates splits.
* The tree structure (feature indices, tree depth, split ordering) is shared with all
  parties, which can reveal information about feature importance and data distribution.
  *Mitigation*: only the feature-owning party recovers the real split threshold; other
  parties store ``NaN`` for that split condition. This prevents non-owners from learning
  the actual split values, while the tree topology and which feature is used at each node
  remain visible. As a consequence, each party's saved model is incomplete for standalone
  inference — collaborative inference across all parties is required (row partitioning
  during training uses ``collective::Allreduce(kBitwiseOR)`` on per-row decision bit
  vectors so that only the feature-owning worker needs the actual split value).

**Security boundary**: HE protects intermediate values in transit and during aggregation.
It does not make the final model universally private — model structure, outputs, and some
aggregated statistics may still reveal information depending on deployment.

Plugin System
=============

To support multiple encryption schemes, XGBoost keeps encryption logic outside the
training core by using a plugin interface. This allows secure schemes to evolve
independently.

Architecture
------------

.. code-block:: none

    +-----------------------------+
    |        XGBoost Core         |
    | histograms / tree building  |
    +--------------+--------------+
                   |
                   v
    +-----------------------------+
    | Federated Processor API     |
    | serialize / invoke plugin   |
    +--------------+--------------+
                   |
                   v
    +-----------------------------+
    | Encryption Handler/Plugin   |
    | Paillier / CKKS / key mgmt  |
    +-----------------------------+

Core Plugin API Surface
-----------------------

The plugin interface is defined in ``plugin/federated/federated_plugin.h``.

**Gradient Encryption (vertical)**

.. code-block:: cpp

    EncryptGradient(float const* in_gpair, size_t n_in,
                    uint8_t** out_gpair, size_t* n_out)

    SyncEncryptedGradient(uint8_t const* in_gpair, size_t n_bytes,
                          uint8_t** out_gpair, size_t* n_out)

**Histogram Encryption (vertical)**

.. code-block:: cpp

    ResetHistContext(uint32_t const* cutptrs, size_t cutptr_len,
                     int32_t const* bin_idx, size_t n_idx)

    BuildEncryptedHistVert(uint64_t const** ridx, size_t const* sizes,
                           int32_t const* nidx, size_t len,
                           uint8_t** out_hist, size_t* out_len)

    SyncEncryptedHistVert(uint8_t* in_hist, size_t len,
                          double** out_hist, size_t* out_len)

**Histogram Encryption (horizontal)**

.. code-block:: cpp

    BuildEncryptedHistHori(double const* in_hist, size_t len,
                           uint8_t** out_hist, size_t* out_len)

    SyncEncryptedHistHori(uint8_t const* in_hist, size_t len,
                          double** out_hist, size_t* out_len)

In this way:

* Encryption library and key handling stay in a separate runtime component.
* XGBoost binary remains lightweight and dependency-minimal.
* Cryptographic backends are easier to rotate or replace.

Implementation Details
======================

With the plugin system, in **secure horizontal FL**, the training loop proceeds as follows:

1. **Sketch synchronization** (once, before training): unchanged —
   ``collective::AllgatherV`` on plaintext sketches (sketches reveal bin boundaries, not
   individual data points).
2. **Local histogram construction** (each boosting round): unchanged — each worker builds
   its local G/H histogram in plaintext using the shared bin boundaries.
3. **Encrypted histogram allreduce** (each boosting round, replaces the plaintext
   ``collective::Allreduce``):

   a. Each worker encrypts its local histogram via the plugin's
      ``BuildEncryptedHistHori``.
   b. Encrypted histograms are sent to the federated server via
      ``collective::AllgatherV``. The server performs homomorphic addition on the
      ciphertexts and returns the aggregated histogram in ciphertext to each worker.
   c. Each worker calls the plugin's ``SyncEncryptedHistHori`` to decrypt the
      aggregated ciphertext, obtaining the global histogram in plaintext.

   In the code this replaces ``AllReduceHist`` with ``AllReduceHistEncrypted``
   (``updater_gpu_hist.cu``), gated by
   ``collective::IsDistributed() && info_.IsRowSplit() && collective::IsEncrypted()``.
4. **Split finding**: unchanged — every worker evaluates splits on the (now-decrypted)
   aggregated histogram independently and arrives at the same best split.
5. **Row partition update**: unchanged — each worker partitions its local rows according
   to the agreed split and proceeds to the next tree node.

In **secure vertical FL**, the training loop proceeds as follows:

1. **Sketch construction** — unchanged: each worker builds sketches for its own features
   locally.
2. **Encrypted gradient broadcast** (replaces plaintext ``collective::Broadcast``): the
   active party (rank 0) computes gradients, encrypts them via the plugin's
   ``EncryptGradient``, then broadcasts ciphertext to all passive parties via
   ``collective::Broadcast``.
3. **Encrypted histogram construction** (replaces plaintext histogram building on passive
   parties): each passive party calls the plugin's ``BuildEncryptedHistVert`` to build
   histograms from the encrypted gradients. Summing encrypted G/H values into histogram
   bins produces valid encrypted bin totals. The active party builds its histogram from
   plaintext gradients as usual.
4. **Gather + decrypt at active party** (replaces ``collective::Allgather`` of split
   candidates): all encrypted histograms are gathered to rank 0 via
   ``collective::AllgatherV``. Rank 0 calls the plugin's ``SyncEncryptedHistVert`` to
   decrypt the aggregated global histogram.
5. **Split finding** (changed — rank 0 only): only the active party evaluates splits on
   the decrypted histogram — passive parties skip split evaluation entirely
   (``is_passive_party = is_col_split_ && is_secure_ && rank != 0``). The best split is
   distributed to all workers via ``collective::Broadcast``.
6. **Split value recovery** (new step): Rank 0 records a bin index rather than an actual
   feature value. Each worker recovers the split value from its own cut-point array —
   only the feature-owning party obtains the real threshold; other parties store ``NaN``.
7. **Tree structure update** — unchanged: all workers apply the chosen split to update
   their row partitions.


**********************************
Secure Federated Learning Examples
**********************************

XGBoost ships with a built-in **mock plugin** (``{"name": "mock"}``) that exercises the
full secure code path — encrypted gradient broadcast, encrypted histogram construction,
gather/decrypt, rank-0-only split evaluation — without performing real encryption (data is
copied as-is). This is useful for validating the pipeline and for testing.

Real Paillier and CKKS implementations are provided by **external libraries** (e.g.,
`NVFlare <https://github.com/NVIDIA/NVFlare>`_) and loaded at runtime via the plugin
system's ``dlopen`` bridge. To use an external plugin, pass ``"path":
"/path/to/plugin.so"`` along with scheme-specific parameters in the ``federated_plugin``
dict.

The examples below use the mock plugin. To enable encryption, pass a ``federated_plugin``
dict.

Horizontal FL (Homomorphic Encryption)
======================================

The horizontal data-loading pattern is unchanged from the non-HE Quick Start (all
parties own labels, ``data_split_mode=0`` by default):

.. code-block:: python

    import xgboost as xgb

    rank = 0

    communicator_env = {
        "dmlc_communicator": "federated",
        "federated_server_address": "localhost:9091",
        "federated_world_size": 3,
        "federated_rank": rank,
        "federated_plugin": {
            "name": "mock",
        },
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        dtrain = xgb.DMatrix(
            f"/tmp/xgboost/federated/horizontal/site-{rank + 1}/train.csv?format=csv&label_column=0"
        )
        dvalid = xgb.DMatrix(
            f"/tmp/xgboost/federated/horizontal/site-{rank + 1}/valid.csv?format=csv&label_column=0"
        )

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": 3,
            "eta": 0.1,
            "nthread": 1,
        }

        bst = xgb.train(
            params, dtrain, num_boost_round=3,
            evals=[(dvalid, "eval"), (dtrain, "train")],
        )

        if xgb.collective.get_rank() == 0:
            bst.save_model("/tmp/xgboost/federated/horizontal_secure_model.json")

Vertical FL (Homomorphic Encryption)
====================================

For vertical FL, the data-loading pattern is unchanged from the non-HE Quick Start
(rank 0 owns labels, ``data_split_mode=1``):

.. code-block:: python

    import xgboost as xgb

    rank = 0

    communicator_env = {
        "dmlc_communicator": "federated",
        "federated_server_address": "localhost:9091",
        "federated_world_size": 3,
        "federated_rank": rank,
        "federated_plugin": {
            "name": "mock",
        },
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        if rank == 0:
            label = "&label_column=0"
        else:
            label = ""

        dtrain = xgb.DMatrix(
            f"/tmp/xgboost/federated/vertical/site-{rank + 1}/train.csv?format=csv{label}",
            data_split_mode=1,
        )
        dvalid = xgb.DMatrix(
            f"/tmp/xgboost/federated/vertical/site-{rank + 1}/valid.csv?format=csv{label}",
            data_split_mode=1,
        )

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": 3,
            "eta": 0.1,
            "nthread": 1,
        }

        bst = xgb.train(
            params, dtrain, num_boost_round=3,
            evals=[(dvalid, "eval"), (dtrain, "train")],
        )

        bst.save_model(f"/tmp/xgboost/federated/vertical_secure_model.{rank}.json")

Comparing Secure Models
========================

With the mock plugin (no real encryption), the secure pipeline produces the same
mathematical result as the non-secure pipeline — the mock plugin copies data as-is, so
gradients and histograms are identical. This means the secure horizontal model should
match the non-secure horizontal model, and the secure vertical model from rank 0 should
match the centralized baseline.

The key difference is in what each party's saved model contains. In secure vertical FL,
only the feature-owning party stores the real split threshold; other parties store
``NaN``. Compare the model JSON files to see this:

.. code-block:: python

    import json

    def load_model_json(path):
        with open(path) as f:
            return json.load(f)

    def get_split_conditions(model_json):
        """Extract split_conditions from all trees."""
        conditions = []
        for tree in model_json["learner"]["gradient_booster"]["model"]["trees"]:
            conditions.append(tree["split_conditions"])
        return conditions

    centralized      = load_model_json("/tmp/xgboost/federated/centralized_model.json")
    horiz_secure     = load_model_json("/tmp/xgboost/federated/horizontal_secure_model.json")
    vert_secure_0    = load_model_json("/tmp/xgboost/federated/vertical_secure_model.0.json")
    vert_secure_1    = load_model_json("/tmp/xgboost/federated/vertical_secure_model.1.json")
    vert_secure_2    = load_model_json("/tmp/xgboost/federated/vertical_secure_model.2.json")

    # Compare model for horizontal FL
    print("\n--- Horizontal FL ---")
    print(f"  secure == centralized : {horiz_secure == centralized}")

    # Compare model for vertical FL
    # Split conditions should align when combining all models.
    print("\n--- Vertical FL ---")
    print("Centralized  :", get_split_conditions(centralized)[0])
    print("Secure rank 0:", get_split_conditions(vert_secure_0)[0])
    print("Secure rank 1:", get_split_conditions(vert_secure_1)[0])
    print("Secure rank 2:", get_split_conditions(vert_secure_2)[0])

    # Combine split conditions across all ranks: for each node, take the
    # non-NaN value (only one rank owns each split feature and stores a real value).
    import math

    all_rank_conds = [
        get_split_conditions(vert_secure_0),
        get_split_conditions(vert_secure_1),
        get_split_conditions(vert_secure_2),
    ]
    combined_conds = []
    for tree_idx in range(len(all_rank_conds[0])):
        merged = []
        for node_idx in range(len(all_rank_conds[0][tree_idx])):
            val = next(
                v for rank_conds in all_rank_conds
                if not math.isnan(v := rank_conds[tree_idx][node_idx])
            )
            merged.append(val)
        combined_conds.append(merged)

    print("\nCombined    :", combined_conds[0])
    print("\n--- Combined == Centralized:", combined_conds == get_split_conditions(centralized))

Each party's model has the correct split conditions only for splits on its own features.
For splits on other parties' features, the split condition is ``NaN``. No single party
can perform standalone inference — all parties must collaborate, combining their local
split knowledge to route each sample through the tree. Combining all models will result in the same
model as centralized training.

**************************
Performance Considerations
**************************

* Homomorphic encryption can dominate runtime cost.
* Ciphertext payloads are significantly larger than plaintext histograms.
* Throughput depends on encryption parameters, network bandwidth, and batching strategy.

Typical tuning levers:

* Reduce communication overhead with batching/compression where supported.
* Use GPU acceleration where encryption backend supports it.
* Calibrate cryptographic parameters with security experts for production targets.

**************
Best Practices
**************

1. Validate end-to-end training first in non-HE mode.
2. Enable TLS before enabling homomorphic encryption.
3. Introduce Paillier (vertical) or CKKS (horizontal) in staged tests.
4. Keep strict key lifecycle controls: generation, storage, rotation, revocation.
5. Monitor convergence and latency together; security knobs can affect both.
6. For horizontal FL, saving the model on rank 0 is sufficient (all ranks produce the
   same model). For vertical FL, every rank must save its own model slice because each
   rank stores only the split values for the features it owns.

***************
Troubleshooting
***************

Workers Cannot Connect
======================

* Confirm server is up before workers start.
* Check host/port and firewall routing.
* Verify every party uses identical ``federated_world_size``.

Training Hangs
==============

* Ensure all ranks joined the communicator context.
* Check data loading path consistency.

TLS Errors
==========

* Validate certificate format (PEM) and file permissions.
* Confirm server cert and client trust chain alignment.
* Use proper CN/SAN values for multi-host deployments.

HE Plugin Errors
================

* Confirm the ``federated_plugin`` dict contains a valid ``name`` field.
* Validate that all plugin-specific keys (e.g. ``key_bits``, ``poly_modulus_degree``) have
  correct types and values.
* Check backend library dependencies and key availability.
* Use ``{"name": "mock"}`` to isolate whether the issue is in the plugin or elsewhere.

***********
Limitations
***********

* Honest-but-curious assumption; no full Byzantine defense.
* Secure aggregation protects intermediates, not all downstream leakage channels.
* No participant dropout support in current design assumptions.
