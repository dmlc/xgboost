############################
Federated Learning with XGBoost
############################

.. contents::
  :local:
  :backlinks: none

**********
Introduction
**********

Federated learning is a machine learning approach that enables training models across multiple parties without sharing raw data. XGBoost's federated learning implementation provides:

* **Privacy Preservation**: Train models without exposing raw data to other parties
* **Secure Aggregation**: Encrypted gradient and histogram synchronization using homomorphic encryption (CKKS, Paillier)
* **Horizontal and Vertical Modes**: Support for data split by rows or by features, following the SecureBoost pattern
* **GPU Acceleration**: CUDA-enabled training for improved performance
* **Plugin Architecture**: Extensible encryption plugin system for custom security schemes

This guide covers how to set up and use federated learning in XGBoost.

.. note:: Security Deep Dive

   For detailed information about security features, encryption schemes, threat models, and the plugin architecture, see :doc:`federated_learning_security`.

**********
Overview
**********

Federated Learning Modes
=========================

XGBoost supports two federated learning paradigms:

**Horizontal Federated Learning**
  Data is partitioned by samples (rows). Each party has the same features but different samples.

  Example: Multiple hospitals training on patient data, each with the same medical measurements but different patients.

**Vertical Federated Learning**
  Data is partitioned by features (columns). Each party has different features for the same samples.

  Example: A bank and an e-commerce company training together, with shared customer IDs but different feature sets (financial vs. shopping behavior).

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

**********
Quick Start
**********

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
   The tree_method must be set to ``hist`` for federated learning. Other tree methods are not supported.

**********
Secure Federated Learning
**********

For production use, enable SSL/TLS encryption to secure communication between server and workers.

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
   For production environments, use properly signed certificates from a trusted Certificate Authority (CA).

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

**********
GPU-Accelerated Federated Learning
**********

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

**********
Complete Example
**********

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

**********
Advanced Configuration
**********

Federated Plugin for Custom Encryption
=======================================

For advanced security requirements, XGBoost supports third-party encryption plugins implementing homomorphic encryption schemes.

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

Higher security parameters (larger polynomial modulus, more coefficient moduli) provide stronger security but:

* Increase computation time (encryption/decryption)
* Increase ciphertext size (network bandwidth)
* Increase memory usage

Consult with cryptography experts to select appropriate parameters for your use case.

Plugin Interface
----------------

The plugin must implement the interface defined in ``plugin/federated/federated_plugin.h``:

* **Gradient Encryption**: ``EncryptGradient()``, ``SyncEncryptedGradient()``
* **Histogram Encryption (Vertical)**: ``BuildEncryptedHistVert()``, ``SyncEncryptedHistVert()``
* **Histogram Encryption (Horizontal)**: ``BuildEncryptedHistHori()``, ``SyncEncryptedHistHori()``

See :doc:`federated_learning_security` for complete plugin interface documentation and security architecture details.

NVFlare Integration
-------------------

For production deployments, use NVIDIA FLARE framework which handles encryption automatically:

.. code-block:: python

    # NVFlare automatically manages encryption
    # No need to manually configure plugins
    from nvflare.app_common.xgb import FedXGBHistogramController

    controller = FedXGBHistogramController(
        num_rounds=100,
        # Encryption processor configuration
        processor_class=XGBHomomorphicEncryptionProcessor,
        processor_args={
            'scheme': 'ckks',
            'poly_modulus_degree': 8192
        }
    )

See `NVFlare documentation <https://nvflare.readthedocs.io/>`_ for details.

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

**********
Troubleshooting
**********

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

**********
Best Practices
**********

1. **Always use SSL/TLS in production** to prevent eavesdropping
2. **Use ``tree_method='hist'``** - it's required for federated learning
3. **Test locally first** with multiple processes before deploying to multiple machines
4. **Monitor training metrics** on all workers to detect data quality issues
5. **Save models only on rank 0** to avoid race conditions
6. **Use proper certificate management** - don't commit certificates to version control
7. **Start with CPU** for debugging, then enable GPU after verifying correctness

**********
Security Model
**********

SecureBoost Pattern (Vertical FL)
==================================

XGBoost's vertical federated learning follows the **SecureBoost** algorithm pattern:

* **Active Party (Server)**: Owns labels and coordinates tree construction
* **Passive Parties (Clients)**: Own feature subsets, perform gradient collection only
* **Label Protection**: Passive parties never see labels, preventing label leakage
* **Distributed Model**: Split values stored distributively to protect feature cut values

During inference, parties collaborate without revealing their feature values to each other.

For details, see :doc:`federated_learning_security` and the `SecureBoost paper <https://arxiv.org/abs/1901.08755>`_.

Homomorphic Encryption (Horizontal FL)
=======================================

XGBoost's horizontal federated learning uses homomorphic encryption to protect histograms:

* **CKKS Encryption**: Local histograms encrypted before transmission
* **Secure Aggregation**: Server aggregates ciphertext, never sees individual histograms
* **Histogram Privacy**: Individual party contributions remain private

**Encryption Overhead:** Training is 2-10x slower due to encryption/decryption operations.

**********
Limitations
**********

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

**********
References
**********

Documentation
=============

* :doc:`federated_learning_security` - Detailed security architecture and threat models
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

External Frameworks
===================

* `NVIDIA FLARE <https://nvflare.readthedocs.io/>`_ - Recommended framework for production deployments
* `Microsoft SEAL <https://github.com/microsoft/SEAL>`_ - CKKS homomorphic encryption library
* `TenSEAL <https://github.com/OpenMined/TenSEAL>`_ - Python wrapper for SEAL

For questions and discussions, visit `XGBoost Discuss Forum <https://discuss.xgboost.ai>`_.
