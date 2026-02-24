##############################
Federated Learning with XGBoost
##############################

Federated learning is a machine learning approach that enables training models across
multiple parties without sharing raw data. XGBoost's federated learning implementation
provides:

* **Privacy Preservation**: Train models without exposing raw data to other parties
* **Secure Aggregation**: Encrypted gradient and histogram synchronization using homomorphic encryption (CKKS, Paillier)
* **Horizontal and Vertical Modes**: Support for data split by rows or by features, following the SecureBoost pattern
* **GPU Acceleration**: CUDA-enabled training for improved performance
* **Plugin Architecture**: Extensible encryption plugin system for custom security schemes

This guide covers how to set up and use federated learning in XGBoost, including security
features, encryption schemes, threat models, and the plugin architecture.

**Contents**

.. contents::
  :backlinks: none
  :local:

********
Overview
********

Federated Learning Modes
=========================

XGBoost supports two federated learning paradigms:

**Horizontal Federated Learning**
  Data is partitioned by samples (rows). Each party has the same features but different
  samples.

  Example: Multiple hospitals training on patient data, each with the same medical
  measurements but different patients.

**Vertical Federated Learning**
  Data is partitioned by features (columns). Each party has different features for the same
  samples.

  Example: A bank and an e-commerce company training together, with shared customer IDs but
  different feature sets (financial vs. shopping behavior).

Architecture
============

A federated learning setup consists of:

1. **Federated Server**: Coordinates training and aggregates encrypted gradients/histograms
2. **Workers (Clients)**: Individual parties that train on their local data
3. **Optional Plugin**: Third-party encryption plugin for secure aggregation

.. code-block:: none

                    ┌─────────────────┐
                    │ Federated Server│
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
      ┌─────▼─────┐    ┌────▼────┐    ┌─────▼─────┐
      │ Worker 0  │    │ Worker 1│    │ Worker 2  │
      │ (Party A) │    │ (Party B)│    │ (Party C) │
      └───────────┘    └─────────┘    └───────────┘

***********
Quick Start
***********

Basic Federated Training (CPU, No SSL)
=======================================

This example demonstrates a simple horizontal federated learning setup with 2 workers.

**Step 1: Start the Federated Server**

.. code-block:: python

    import xgboost.federated

    # Start server (blocking mode)
    xgboost.federated.run_federated_server(
        n_workers=2,
        port=9091
    )

**Step 2: Start Workers (in separate processes/machines)**

.. code-block:: python

    import xgboost as xgb

    # Worker configuration
    communicator_env = {
        'dmlc_communicator': 'federated',
        'federated_server_address': 'localhost:9091',
        'federated_world_size': 2,
        'federated_rank': 0,  # Change to 1 for second worker
    }

    # Initialize communicator context
    with xgb.collective.CommunicatorContext(**communicator_env):
        # Load local data
        dtrain = xgb.DMatrix('worker0_train.txt')
        dtest = xgb.DMatrix('worker0_test.txt')

        # Standard XGBoost training
        params = {
            'max_depth': 3,
            'eta': 0.3,
            'objective': 'binary:logistic',
            'tree_method': 'hist',  # Required for federated learning
        }

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=10,
            evals=[(dtest, 'test')]
        )

        # Save model (only on rank 0)
        if xgb.collective.get_rank() == 0:
            bst.save_model('federated_model.json')

**Step 3: Run Everything**

You'll need to:

1. Start the server script
2. Start worker scripts (one for each party)
3. Wait for training to complete

.. note::
   The tree_method must be set to ``hist`` for federated learning. Other tree methods are
   not supported.

***************************
Secure Federated Learning
***************************

For production use, enable SSL/TLS encryption to secure communication between server and
workers.

Generate SSL Certificates
==========================

For testing, generate self-signed certificates:

.. code-block:: bash

    # Generate server certificate
    openssl req -x509 -newkey rsa:2048 -days 365 -nodes \
        -keyout server-key.pem -out server-cert.pem \
        -subj "/C=US/CN=localhost"

    # Generate client certificate
    openssl req -x509 -newkey rsa:2048 -days 365 -nodes \
        -keyout client-key.pem -out client-cert.pem \
        -subj "/C=US/CN=localhost"

.. warning::
   For production environments, use properly signed certificates from a trusted Certificate
   Authority (CA).

Server with SSL
===============

.. code-block:: python

    import xgboost.federated

    xgboost.federated.run_federated_server(
        n_workers=2,
        port=9091,
        server_key_path='server-key.pem',
        server_cert_path='server-cert.pem',
        client_cert_path='client-cert.pem',
    )

Worker with SSL
===============

.. code-block:: python

    import xgboost as xgb

    communicator_env = {
        'dmlc_communicator': 'federated',
        'federated_server_address': 'localhost:9091',
        'federated_world_size': 2,
        'federated_rank': 0,
        # SSL configuration
        'federated_server_cert_path': 'server-cert.pem',
        'federated_client_key_path': 'client-key.pem',
        'federated_client_cert_path': 'client-cert.pem',
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        # Training code same as before
        dtrain = xgb.DMatrix('data.txt')
        bst = xgb.train({'tree_method': 'hist'}, dtrain, num_boost_round=10)

************************************
GPU-Accelerated Federated Learning
************************************

Enable GPU acceleration for faster training on large datasets.

.. code-block:: python

    import xgboost as xgb

    communicator_env = {
        'dmlc_communicator': 'federated',
        'federated_server_address': 'localhost:9091',
        'federated_world_size': 2,
        'federated_rank': 0,
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        dtrain = xgb.DMatrix('data.txt')

        params = {
            'tree_method': 'hist',
            'device': 'cuda:0',  # Use GPU 0
            'max_depth': 5,
            'eta': 0.3,
        }

        bst = xgb.train(params, dtrain, num_boost_round=100)

**Multi-GPU Setup**

For workers with multiple GPUs, assign different devices:

.. code-block:: python

    # Worker 0 uses GPU 0
    device = f'cuda:{rank}'  # If rank=0, device='cuda:0'

    params = {
        'tree_method': 'hist',
        'device': device,
        # ... other params
    }

****************
Complete Example
****************

Here's a complete, production-ready example using multiprocessing:

.. code-block:: python

    import multiprocessing
    import time
    import xgboost as xgb
    import xgboost.federated


    def run_server(n_workers, port):
        """Run federated server."""
        xgboost.federated.run_federated_server(
            n_workers=n_workers,
            port=port,
            server_key_path='server-key.pem',
            server_cert_path='server-cert.pem',
            client_cert_path='client-cert.pem',
        )


    def run_worker(rank, world_size, port):
        """Run federated worker."""
        communicator_env = {
            'dmlc_communicator': 'federated',
            'federated_server_address': f'localhost:{port}',
            'federated_world_size': world_size,
            'federated_rank': rank,
            'federated_server_cert_path': 'server-cert.pem',
            'federated_client_key_path': 'client-key.pem',
            'federated_client_cert_path': 'client-cert.pem',
        }

        with xgb.collective.CommunicatorContext(**communicator_env):
            # Load worker-specific data
            dtrain = xgb.DMatrix(f'worker_{rank}_train.txt')
            dtest = xgb.DMatrix(f'worker_{rank}_test.txt')

            params = {
                'max_depth': 5,
                'eta': 0.3,
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'eval_metric': 'logloss',
            }

            results = {}
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=50,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                evals_result=results,
                verbose_eval=True,
            )

            # Only rank 0 saves the model
            if xgb.collective.get_rank() == 0:
                bst.save_model('federated_model.json')
                print(f"Model saved. Final test logloss: {results['test']['logloss'][-1]:.4f}")


    def main():
        """Main orchestration function."""
        world_size = 3
        port = 9091

        # Start server
        server = multiprocessing.Process(target=run_server, args=(world_size, port))
        server.start()
        time.sleep(2)  # Give server time to start

        # Start workers
        workers = []
        for rank in range(world_size):
            worker = multiprocessing.Process(target=run_worker, args=(rank, world_size, port))
            workers.append(worker)
            worker.start()

        # Wait for completion
        for worker in workers:
            worker.join()

        server.terminate()
        print("Federated training completed successfully!")


    if __name__ == '__main__':
        main()

**************
Security Model
**************

XGBoost's secure federated learning implementation protects data privacy during distributed
training across multiple parties. The security model is based on research proposals
documented in:

- `RFC #9987: Secure Vertical Federated Learning <https://github.com/dmlc/xgboost/issues/9987>`_
- `RFC #10170: Secure Horizontal Federated Learning <https://github.com/dmlc/xgboost/issues/10170>`_

Threat Model
=============

Trust Assumptions
-----------------

The secure federated learning implementation assumes:

* **Trusted Partners**: A few trusted partners jointly train a model
* **Honest-but-Curious Parties**: Participants follow the protocol but may attempt to learn information from received data
* **Pre-Established Relationships**: Private Set Intersection (PSI) is completed before training
* **Network Security**: Fast, secure network connectivity between participants and central server
* **No Participant Dropout**: All parties remain available throughout training (dropout support is a non-goal)

Security Boundaries
-------------------

**What is Protected:**

* **Label Information**: Server/active party labels are protected from passive clients (vertical FL)
* **Feature Values**: Individual feature values not leaked between parties
* **Histograms**: Local gradient/Hessian histograms encrypted before transmission
* **Gradients**: Local gradients encrypted to prevent information leakage

**What is NOT Protected:**

* **Model Structure**: The final tree structure is shared among all parties
* **Aggregated Statistics**: Final aggregated histograms are revealed (in decrypted form)
* **Participation Patterns**: Which parties participate in training is observable

SecureBoost Pattern (Vertical Federated Learning)
==================================================

The vertical federated learning implementation follows a variation of the
`SecureBoost algorithm <https://arxiv.org/abs/1901.08755>`_:

* **Active Party (Server)**: Owns labels and coordinates tree construction
* **Passive Parties (Clients)**: Own feature subsets, perform gradient collection only
* **Label Protection**: Passive parties never see labels, preventing label leakage
* **Distributed Model**: Split values stored distributively to protect feature cut values

Key Architectural Change
------------------------

**Traditional Vertical FL:**

.. code-block:: none

    Client: [Data] → [Gradient Computation] → [Histogram] → [Tree Construction]
    Server: [Labels] → [Loss Computation]

**SecureBoost Pattern:**

.. code-block:: none

    Client (Passive): [Features] → [Gradient Collection Only]
    Server (Active): [Labels + Features] → [Full Tree Construction]

This architectural change:

* Moves tree construction to the server (active party)
* Clients only perform gradient collection
* Server performs histogram synchronization and split finding
* Protects label information from passive clients

Secure Inference
----------------

Model storage is distributed to protect feature cut values:

* **Feature-Owning Party**: Stores actual split values for their features
* **Other Parties**: Store ``NaN`` for splits they don't own
* **Prediction Time**: Parties collaborate, each evaluating splits for their features only

This prevents feature value leakage even after model deployment.

**Example:**

.. code-block:: python

    # Party A's view of the model (owns features 0-5)
    Split: feature_0 < 0.5  → [actual value stored]
    Split: feature_8 < 1.2  → [NaN stored, Party B owns this]

    # Party B's view of the model (owns features 6-10)
    Split: feature_0 < 0.5  → [NaN stored, Party A owns this]
    Split: feature_8 < 1.2  → [actual value stored]

Homomorphic Encryption (Horizontal Federated Learning)
=======================================================

The horizontal federated learning implementation uses homomorphic encryption to protect
histogram data.

CKKS Encryption Scheme
-----------------------

**Selected Approach:** CKKS (Cheon-Kim-Kim-Song) homomorphic encryption

**Why CKKS:**

* Horizontal FL requires light vector additions across parties
* CKKS supports addition and multiplication on encrypted data
* Efficient for approximate arithmetic on real numbers
* Well-suited for histogram aggregation (sum of G/H values)

**Alternatives Considered:**

* **Paillier**: Supports only addition, but simpler
* **BFV/BGV**: Integer-based, less suitable for floating-point histograms

**Limitations:**

* Cannot support division or argmax operations in encrypted space
* Tree construction must occur in plaintext (on aggregated histograms)

Encryption Data Flow
--------------------

**Horizontal Federated Learning with CKKS:**

.. code-block:: none

    1. Local Computation (Each Party):
       Data → Histogram (G/H pairs) → [PLAINTEXT]

    2. Encryption (Each Party):
       Histogram → Encrypt(CKKS) → [CIPHERTEXT]

    3. Transmission:
       Party → Server: [CIPHERTEXT only]

    4. Aggregation (Server):
       Sum([CIPHERTEXT₁, CIPHERTEXT₂, ...]) → [AGGREGATED_CIPHERTEXT]

    5. Decryption (Distributed or Server):
       [AGGREGATED_CIPHERTEXT] → Decrypt → [PLAINTEXT_AGGREGATED]

    6. Tree Construction:
       [PLAINTEXT_AGGREGATED] → Find Best Split → Update Tree

**Key Security Property:** Individual party histograms never leave their local environment
in plaintext form.

Security Guarantees
====================

Vertical Federated Learning
----------------------------

**What is Protected:**

1. **Label Privacy**: Passive parties cannot access labels

   * Labels remain on active party (server)
   * Only encrypted gradients shared

2. **Feature Value Privacy**: Feature cut values are distributed

   * Each party stores only their own feature splits
   * Other splits stored as NaN
   * Collaborative inference required

3. **Histogram Privacy**: Local histograms encrypted before sharing

   * Cumulative histogram exposure minimized
   * Only aggregated histogram revealed after decryption

**Attack Resistance:**

* **Gradient Inversion Attacks**: Gradients encrypted, cannot be directly inverted
* **Feature Reconstruction**: Distributed split storage prevents feature value leakage
* **Model Stealing**: Partial model at each party, full model not recoverable by any single party

Horizontal Federated Learning
-------------------------------

**What is Protected:**

1. **Histogram Privacy**: Local G/H histograms never transmitted in plaintext

   * Encrypted with CKKS before leaving local environment
   * Server sees only aggregated ciphertext

2. **Sample-Level Privacy**: Individual sample contributions hidden

   * Aggregation in ciphertext prevents isolation of individual contributions
   * Server cannot decrypt individual party histograms

3. **Gradient Privacy**: Similar protections apply to gradients

**Attack Resistance:**

* **Histogram Leakage**: All transmission in encrypted form
* **Membership Inference**: Aggregation masks individual contributions
* **Byzantine Attacks**: Honest-but-curious model assumed (malicious parties not defended against)

**Known Limitations:**

* **Aggregated Information Leakage**: Final aggregated histogram is revealed
* **Collusion**: Multiple colluding parties could potentially reconstruct more information
* **Malicious Parties**: Current design does not defend against actively malicious participants

*******************
Plugin Architecture
*******************

Processor Interface Pattern
============================

The security implementation uses a **processor interface** to decouple encryption from
XGBoost core:

.. code-block:: none

    ┌─────────────────────────────────────────────────────┐
    │                   XGBoost Core                      │
    │  (Histogram Computation, Tree Construction)         │
    └────────────────┬────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────┐
    │            Processor Interface (Plugin)             │
    │  - Serialize histogram data                         │
    │  - Call encryption handler                          │
    │  - Deserialize encrypted/decrypted data             │
    └────────────────┬────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────┐
    │         gRPC Handler (External Process)             │
    │  - Paillier encryption (Python/C++ library)         │
    │  - CKKS encryption (SEAL, TenSEAL, etc.)           │
    │  - Key management                                   │
    └─────────────────────────────────────────────────────┘

**Benefits:**

* XGBoost remains dependency-free (no encryption library dependencies)
* Flexible encryption scheme selection
* Easy to swap encryption implementations
* External security audit of encryption components

Plugin Interface API
=====================

The plugin interface is defined in ``plugin/federated/federated_plugin.h``:

Gradient Encryption (Vertical FL)
----------------------------------

.. code-block:: cpp

    // Encrypt gradients on active party
    EncryptGradient(float const* in_gpair, size_t n_in,
                    uint8_t** out_gpair, size_t* n_out)

    // Synchronize encrypted gradients after broadcast
    SyncEncryptedGradient(uint8_t const* in_gpair, size_t n_bytes,
                          uint8_t** out_gpair, size_t* n_out)

Histogram Encryption (Vertical FL)
-----------------------------------

.. code-block:: cpp

    // Set context for vertical histogram building
    ResetHistContext(uint32_t const* cutptrs, size_t cutptr_len,
                     int32_t const* bin_idx, size_t n_idx)

    // Build encrypted histogram for local features
    BuildEncryptedHistVert(uint64_t const** ridx, size_t const* sizes,
                           int32_t const* nidx, size_t len,
                           uint8_t** out_hist, size_t* out_len)

    // Synchronize and decrypt aggregated histogram
    SyncEncryptedHistVert(uint8_t* in_hist, size_t len,
                          double** out_hist, size_t* out_len)

Histogram Encryption (Horizontal FL)
-------------------------------------

.. code-block:: cpp

    // Encrypt local histogram
    BuildEncryptedHistHori(double const* in_hist, size_t len,
                           uint8_t** out_hist, size_t* out_len)

    // Aggregate and decrypt histograms
    SyncEncryptedHistHori(uint8_t const* in_hist, size_t len,
                          double** out_hist, size_t* out_len)

Two-Tiered Encryption Approaches
=================================

The architecture supports two implementation strategies:

Option 1: Handler-Side Encryption (Recommended)
------------------------------------------------

* gRPC local handler performs encryption/decryption
* Uses external Python/C++ encryption libraries (e.g., ``tenseal``, ``python-paillier``)
* XGBoost communicates via processor interface
* No encryption dependencies in XGBoost binary

**Advantages:**

* Keeps XGBoost dependency-free
* Easy to update encryption libraries
* Multiple encryption schemes supported without rebuilding XGBoost
* Separate security audit scope

Option 2: XGBoost-Side Encryption
----------------------------------

* Direct C++ encryption integration (e.g., Microsoft SEAL for CKKS)
* Encryption built into XGBoost binary
* Plugin loaded via ``dlopen`` at runtime

**Advantages:**

* More efficient (no IPC overhead)
* Tighter integration
* Better performance for encryption-heavy operations

**Disadvantages:**

* Larger binary size
* Build complexity
* Harder to update encryption schemes

**Configuration Example:**

.. code-block:: python

    communicator_env = {
        'dmlc_communicator': 'federated',
        'federated_server_address': 'localhost:9091',
        'federated_world_size': 3,
        'federated_rank': 0,
        # Load encryption plugin
        'federated_plugin_path': '/usr/lib/xgboost/libckks_plugin.so',
        'federated_plugin_config': '{"scheme": "ckks", "poly_modulus": 8192}'
    }

**********************
Advanced Configuration
**********************

Federated Plugin for Custom Encryption
=======================================

For advanced security requirements, XGBoost supports third-party encryption plugins
implementing homomorphic encryption schemes.

Supported Encryption Schemes
-----------------------------

The plugin architecture supports multiple encryption schemes:

* **CKKS (Recommended for Horizontal FL)**: Approximate homomorphic encryption for real numbers

  - Supports addition and multiplication on encrypted data
  - Efficient for histogram aggregation across parties
  - Implementations: Microsoft SEAL, TenSEAL

* **Paillier (Suitable for Vertical FL)**: Additive homomorphic encryption

  - Supports only addition on encrypted data
  - Simpler than CKKS, but limited operations
  - Implementations: python-paillier, libpaillier

* **BFV/BGV**: Integer-based homomorphic encryption

  - Alternative to CKKS for integer arithmetic
  - Less suitable for floating-point histograms

Plugin Configuration
--------------------

**Example with Custom Encryption Plugin:**

.. code-block:: python

    communicator_env = {
        'dmlc_communicator': 'federated',
        'federated_server_address': 'localhost:9091',
        'federated_world_size': 2,
        'federated_rank': 0,
        # Plugin configuration
        'federated_plugin': {
            'path': '/path/to/ckks_plugin.so',
            'config': {
                'scheme': 'ckks',
                'poly_modulus_degree': 8192,
                'coeff_mod_bit_sizes': [60, 40, 40, 60],
                'scale': 2**40
            }
        }
    }

**Security-Performance Trade-off:**

Higher security parameters (larger polynomial modulus, more coefficient moduli) provide
stronger security but:

* Increase computation time (encryption/decryption)
* Increase ciphertext size (network bandwidth)
* Increase memory usage

Consult with cryptography experts to select appropriate parameters for your use case.

NVFlare Integration
====================

NVIDIA FLARE (Federated Learning Application Runtime Environment) provides the recommended
framework for production secure federated learning deployments. NVFlare automatically
manages encryption via the processor interface:

.. code-block:: python

    # NVFlare server configuration
    from nvflare.app_common.xgb import FedXGBHistogramController

    controller = FedXGBHistogramController(
        num_rounds=100,
        early_stopping_rounds=10,
        processor_class=XGBHomomorphicEncryptionProcessor,
        processor_args={
            'scheme': 'ckks',
            'poly_modulus_degree': 8192,
            'coeff_mod_bit_sizes': [60, 40, 40, 60]
        }
    )

    # NVFlare client configuration
    from nvflare.app_common.xgb import FedXGBHistogramExecutor

    executor = FedXGBHistogramExecutor(
        data_loader_id="data_loader",
        world_size=5,
        rank=0
    )

See `NVFlare XGBoost Documentation <https://nvflare.readthedocs.io/en/main/examples/xgboost.html>`_
for complete examples.

Timeout Configuration
=====================

Adjust connection timeout if workers need more time to connect:

.. code-block:: python

    xgboost.federated.run_federated_server(
        n_workers=10,
        port=9091,
        timeout=600,  # 10 minutes
    )

Non-Blocking Server Mode
=========================

For custom orchestration, run the server in non-blocking mode:

.. code-block:: python

    # Returns worker arguments immediately
    worker_args = xgboost.federated.run_federated_server(
        n_workers=2,
        port=9091,
        blocking=False,
    )

    # Use worker_args to configure workers programmatically
    # The server runs in a background thread

**************************
Performance Considerations
**************************

Encryption Overhead
===================

**Typical Overhead:**

* **CKKS Encryption**: 10-100x slower than plaintext operations (depends on security parameters)
* **Paillier Encryption**: 100-1000x slower (depends on key size)
* **Network**: Encrypted data is larger (ciphertext expansion)

**Optimization Strategies:**

* Use GPU acceleration for encryption/decryption
* Batch encryption operations
* Optimize security parameters (polynomial modulus, key size) for use case
* Use efficient serialization formats

Communication Costs
===================

**Plaintext vs Ciphertext:**

* **Plaintext Histogram**: 2 × n_bins × sizeof(double) bytes per histogram
* **CKKS Ciphertext**: ~10-50x larger (depends on parameters)
* **Network Bandwidth**: Can become bottleneck with many parties

**Mitigation:**

* Compression of ciphertext
* Efficient allgather implementations
* Network topology optimization

**************
Best Practices
**************

Security Configuration
======================

1. **Always Use SSL/TLS**: Even with encryption, use SSL/TLS for transport security
2. **Key Management**: Use proper key generation and storage

   * Generate fresh keys per training session
   * Use hardware security modules (HSM) for production
   * Implement key rotation policies

3. **Parameter Selection**: Choose encryption parameters carefully

   * Higher security = more computation + larger ciphertext
   * Balance security needs with performance
   * Consult cryptography experts for production deployments

4. **Audit Encryption Plugins**: Independently audit third-party encryption plugins
5. **Monitor for Anomalies**: Log and monitor federated training for unusual patterns

Training Best Practices
=======================

1. **Use ``tree_method='hist'``** - it's required for federated learning
2. **Test locally first** with multiple processes before deploying to multiple machines
3. **Monitor training metrics** on all workers to detect data quality issues
4. **Save models only on rank 0** to avoid race conditions
5. **Start with CPU** for debugging, then enable GPU after verifying correctness
6. **Use proper certificate management** - don't commit certificates to version control

Privacy Risk Assessment
=======================

Before deploying secure federated learning:

1. **Threat Modeling**: Identify your specific adversaries and threats
2. **Data Sensitivity**: Assess sensitivity of labels, features, and aggregates
3. **Regulatory Compliance**: Ensure compliance with GDPR, HIPAA, etc.
4. **Differential Privacy**: Consider adding differential privacy for stronger guarantees
5. **Secure Computation Limitations**: Understand what is and isn't protected

***************
Troubleshooting
***************

Connection Issues
=================

**Problem:** Workers can't connect to server

**Solutions:**

* Verify server is running before starting workers
* Check firewall rules allow connections on the specified port
* Ensure ``federated_server_address`` is correct (use actual hostname/IP, not ``localhost``, for multi-machine setups)
* Increase timeout value if workers are slow to start

SSL Certificate Errors
=======================

**Problem:** SSL handshake failures

**Solutions:**

* Ensure all certificate files are in the correct PEM format
* Verify certificate paths are correct and readable
* Check that server and client certificates match
* For multi-machine setups, update certificate CN (Common Name) to match server hostname

Training Hangs
==============

**Problem:** Training starts but never completes

**Solutions:**

* Ensure all workers have connected (check server logs)
* Verify ``world_size`` matches actual number of workers
* Check that all workers are using ``tree_method='hist'``
* Look for errors in worker logs

GPU Out of Memory
=================

**Problem:** CUDA out of memory errors

**Solutions:**

* Reduce ``max_bin`` parameter to use less memory
* Decrease ``max_depth``
* Use smaller batch sizes if training incrementally
* Assign workers to different GPUs to distribute memory load

***********
Limitations
***********

Security Limitations
====================

* **Honest-but-Curious Model**: Assumes parties follow protocol but may observe data
* **No Byzantine Fault Tolerance**: Does not protect against actively malicious parties
* **Aggregated Information Leakage**: Final aggregated histograms are revealed in plaintext
* **No Differential Privacy**: Does not provide formal differential privacy guarantees (can be added separately)
* **Collusion Risk**: Multiple colluding parties could potentially reconstruct more information

Functional Limitations
======================

* Only ``hist`` tree method is supported (not ``approx`` or ``exact``)

  - Homomorphic encryption schemes don't support division/argmax operations
  - Tree construction must occur on plaintext aggregated histograms

* All workers must use the same XGBoost version
* Data schema (number and order of features) must match across workers for horizontal learning
* Sample IDs must match across workers for vertical learning
* Network latency affects training speed - federated learning is slower than centralized training
* No support for participant dropout during training

********************
Implementation Phases
********************

The secure federated learning feature was developed in phases:

Phase 1: Vertical Pipeline Foundation
======================================

* Alternative vertical pipeline with histogram synchronization
* Basic infrastructure for column-split data
* PR: `#10037 <https://github.com/dmlc/xgboost/pull/10037>`_

Phase 2: Processor Interface
=============================

* Plugin architecture for encryption
* Processor interface definition
* Integration with NVFlare
* PRs: `#10124 <https://github.com/dmlc/xgboost/pull/10124>`_, `#10231 <https://github.com/dmlc/xgboost/pull/10231>`_

Phase 3: Secure Evaluation
===========================

* Distributed model storage
* Secure inference without feature leakage
* Validation/evaluation with privacy preservation
* PR: `#10079 <https://github.com/dmlc/xgboost/pull/10079>`_

Phase 4: GPU Acceleration
==========================

* CUDA-accelerated vertical federated scheme
* GPU histogram computation
* GPU-based encryption/decryption
* PR: `#10652 <https://github.com/dmlc/xgboost/pull/10652>`_

Phase 5: Horizontal Federated Learning
=======================================

* CKKS-based horizontal FL
* Secure histogram aggregation
* CPU and GPU support
* PR: `#10601 <https://github.com/dmlc/xgboost/pull/10601>`_

***********************
Testing Security Features
***********************

Unit Tests
==========

Security features include comprehensive tests:

* **C++ Tests**: ``tests/cpp/plugin/federated/test_federated_plugin.cc``
* **Python Tests**: ``tests/test_distributed/test_federated/``
* **GPU Tests**: ``tests/test_distributed/test_gpu_federated/``

Running Security Tests
======================

.. code-block:: bash

    # Test federated plugin interface
    cd build
    ./testxgboost --gtest_filter="*Federated*"

    # Test Python federated learning
    export PYTHONPATH=./python-package
    pytest -v tests/test_distributed/test_federated/

    # Test with encryption (requires plugin)
    export FEDERATED_PLUGIN_PATH=/path/to/encryption_plugin.so
    pytest -v tests/test_distributed/test_federated/ --secure

***
FAQ
***

**Q: Is the model trained with secure FL identical to centralized training?**

A: Yes, the final model is mathematically equivalent. The encryption only protects
intermediate values (gradients, histograms) during transmission.

**Q: Can I use secure FL without a framework like NVFlare?**

A: Yes, but you'll need to implement the encryption plugin yourself following the interface
in ``federated_plugin.h``.

**Q: What encryption scheme should I use?**

A: For horizontal FL, CKKS is recommended. For vertical FL, Paillier or CKKS both work.
Consult a cryptography expert for production.

**Q: Is secure FL slower than regular FL?**

A: Yes, encryption adds overhead. Expect 2-10x slower depending on security parameters and
hardware acceleration.

**Q: Does secure FL protect against malicious parties?**

A: No, the current design assumes honest-but-curious parties. Byzantine fault tolerance is
a non-goal.

**Q: Can I add differential privacy on top of secure FL?**

A: Yes, differential privacy can be added independently to provide additional privacy
guarantees.

**********
References
**********

Documentation
=============

* :doc:`/parameter` - Complete parameter reference
* :doc:`/python/python_api` - Python API documentation
* `Federated Plugin Interface <https://github.com/dmlc/xgboost/blob/master/plugin/federated/federated_plugin.h>`_ - C++ plugin specification

RFCs and Design Documents
==========================

* `RFC #9987: Secure Vertical Federated Learning <https://github.com/dmlc/xgboost/issues/9987>`_ - SecureBoost pattern, processor interface
* `RFC #10170: Secure Horizontal Federated Learning <https://github.com/dmlc/xgboost/issues/10170>`_ - CKKS encryption, threat model

Academic Papers
===============

* `SecureBoost: A Lossless Federated Learning Framework <https://arxiv.org/abs/1901.08755>`_ - Vertical FL algorithm
* `CKKS: Homomorphic Encryption for Arithmetic of Approximate Numbers <https://eprint.iacr.org/2016/421>`_ - CKKS scheme
* `Practical Secure Aggregation for Privacy-Preserving Machine Learning <https://eprint.iacr.org/2017/281>`_ - Secure aggregation

Libraries and Tools
===================

* `Microsoft SEAL <https://github.com/microsoft/SEAL>`_ - CKKS implementation
* `TenSEAL <https://github.com/OpenMined/TenSEAL>`_ - Python SEAL wrapper
* `python-paillier <https://github.com/data61/python-paillier>`_ - Paillier encryption
* `NVIDIA FLARE <https://github.com/NVIDIA/NVFlare>`_ - Recommended framework for production deployments

For questions and discussions, visit the `XGBoost Discussion Forum <https://discuss.xgboost.ai>`_.
