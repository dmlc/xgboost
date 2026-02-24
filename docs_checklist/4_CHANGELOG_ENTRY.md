# Change Log Entry for Federated Learning

## Target Release
This change should be documented in the next XGBoost release (v2.2.0 or later, depending on when it's merged).

---

## Section to Add to Release Notes

### Location
Add as a **major section** in the release notes, after "Networking Improvements" or similar infrastructure sections.

### Content

```rst
*********************************************
Secure Federated Learning with GPU Support
*********************************************

This release introduces comprehensive secure federated learning capabilities, enabling privacy-preserving distributed training across multiple parties without sharing raw data. This major feature includes:

**Core Features:**

- **Horizontal Federated Learning**: Train models where data is partitioned by samples (rows). Each party has the same features but different training instances.
- **Vertical Federated Learning**: Train models where data is partitioned by features (columns). Each party has different feature sets for the same samples.
- **GPU Acceleration**: Full CUDA support for both horizontal and vertical federated schemes, providing significant speedup for large-scale federated training.
- **Secure Aggregation**: Encrypted gradient and histogram synchronization to protect data privacy during training.
- **Plugin Architecture**: Extensible plugin system for third-party encryption providers to implement custom secure aggregation protocols.

**API Additions:**

- New Python module: ``xgboost.federated`` providing:

  * ``FederatedTracker``: Manages federated learning coordination
  * ``run_federated_server()``: Convenience function to start federated servers

- Worker-side configuration via ``xgboost.collective.CommunicatorContext`` with new parameters:

  * ``dmlc_communicator='federated'``: Enable federated mode
  * ``federated_server_address``: Server connection endpoint
  * ``federated_world_size``: Number of participating workers
  * ``federated_rank``: Unique worker identifier
  * ``federated_server_cert_path``, ``federated_client_key_path``, ``federated_client_cert_path``: SSL/TLS encryption support

**Security:**

- SSL/TLS encryption for secure communication between server and workers
- Optional third-party plugin support for advanced encryption schemes (homomorphic encryption, secure multi-party computation)
- Protection against gradient leakage in both horizontal and vertical scenarios

**Performance:**

- GPU-accelerated histogram computation and communication
- Efficient gradient synchronization using collective operations
- Support for NCCL-based GPU-to-GPU communication

**Testing & Documentation:**

- Comprehensive test suite covering:

  * CPU and GPU federated training
  * Horizontal and vertical learning modes
  * Secure (SSL) and non-secure modes
  * Multi-worker scenarios

- New tutorial: :doc:`/tutorials/federated_learning` with complete usage examples
- Full parameter documentation in :doc:`/parameter`
- API documentation for Python, C++, and plugin interfaces

**Usage Example:**

.. code-block:: python

    import xgboost as xgb
    import xgboost.federated

    # Start federated server (in a separate process)
    xgboost.federated.run_federated_server(
        n_workers=3,
        port=9091,
        server_key_path='server-key.pem',
        server_cert_path='server-cert.pem',
        client_cert_path='client-cert.pem',
    )

    # Worker-side training
    communicator_env = {
        'dmlc_communicator': 'federated',
        'federated_server_address': 'localhost:9091',
        'federated_world_size': 3,
        'federated_rank': 0,  # Worker 0
        'federated_server_cert_path': 'server-cert.pem',
        'federated_client_key_path': 'client-key.pem',
        'federated_client_cert_path': 'client-cert.pem',
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        dtrain = xgb.DMatrix('local_data.txt')
        params = {
            'tree_method': 'hist',  # Required
            'device': 'cuda:0',     # GPU acceleration
            'max_depth': 5,
            'eta': 0.3,
        }
        bst = xgb.train(params, dtrain, num_boost_round=100)

**Implementation Details:**

- Plugin architecture defined in ``plugin/federated/federated_plugin.h``
- Histogram synchronization via ``plugin/federated/federated_hist.{cc,h}``
- Secure collective operations in ``plugin/federated/federated_coll.{cc,cu}``
- Communication layer in ``plugin/federated/federated_comm.{cc,cu}``

**Related PRs:** (#10652, #10671, #10675, #10601, #10629, #10621, #10622, #10534, #10589, #10569, #10565, #10542, #10540, #10530, #10528, #10079, #10037)

**Acknowledgments:**

This feature was developed in collaboration with the secure federated learning community and represents a significant milestone in privacy-preserving machine learning with XGBoost.

**Notes:**

- Federated learning requires ``tree_method='hist'``
- Currently only Python bindings are available; JVM and R support planned for future releases
- Network latency affects training speed; federated learning is inherently slower than centralized training
- For production deployments, always use SSL/TLS encryption
```

---

## Alternative: Shorter Version (if space is limited)

If the release notes need to be more concise, use this shorter version:

```rst
******************
Federated Learning
******************

This release adds comprehensive secure federated learning support for privacy-preserving distributed training:

- **Horizontal and Vertical Federated Learning**: Support for both sample-split and feature-split scenarios
- **GPU Acceleration**: Full CUDA support for federated histogram computation and gradient aggregation
- **Secure Communication**: SSL/TLS encryption and plugin architecture for custom secure aggregation
- **New API**: ``xgboost.federated`` module with ``FederatedTracker`` and ``run_federated_server()``
- **Documentation**: Complete tutorial at :doc:`/tutorials/federated_learning`

Usage requires setting ``tree_method='hist'`` and configuring workers via ``xgboost.collective.CommunicatorContext`` with ``dmlc_communicator='federated'``. See the federated learning tutorial for complete examples.

**Related PRs:** (#10652, #10671, #10675, #10601, #10629, #10621, #10622, #10534, #10589, #10569, #10565, #10542, #10540, #10530, #10528)
```

---

## Additional Section: Breaking Changes (if any)

If there are any breaking changes related to federated learning, add to the "Breaking Changes" section:

```rst
*******************
Breaking Changes
*******************

Federated Learning:

- The ``manylinux2014`` Python wheel variant does not support federated learning. Users on older Linux distributions must upgrade to glibc 2.28+ or build from source. (Already mentioned in v2.1.0, reiterated here for clarity)
- Federated learning is only compatible with ``tree_method='hist'``. Attempting to use other tree methods will raise an error.
```

---

## Additional Section: Python Package Changes

If there's a dedicated Python section in the release notes, add:

```rst
****************
Python Package
****************

New Features:

- New module ``xgboost.federated`` for federated learning:

  * ``FederatedTracker``: Class for managing federated training coordination
  * ``run_federated_server()``: Convenience function to start federated servers with SSL/TLS support

- Extended ``xgboost.collective.CommunicatorContext`` with federated learning parameters
- New testing utilities in ``xgboost.testing.federated`` for federated learning tests

Documentation:

- Comprehensive federated learning tutorial with examples
- API documentation for all federated learning classes and functions
```

---

## Git Commit Message for Change Log

When committing the change log update:

```
[doc] Add release notes for secure federated learning

- Document horizontal and vertical federated learning support
- Highlight GPU acceleration and secure aggregation features
- Include API changes and usage examples
- List related PRs and acknowledgments

Related: #XXXXX (PR number)
```

---

## Verification Checklist

After adding the change log entry:

- [ ] Entry is in the correct release notes file (e.g., ``doc/changes/v2.2.0.rst``)
- [ ] File is listed in ``doc/changes/index.rst``
- [ ] All PR numbers are correct and formatted as ``(#XXXXX)``
- [ ] Cross-references to documentation use correct paths (e.g., ``:doc:`/tutorials/federated_learning```)
- [ ] Code examples are properly formatted with ``.. code-block:: python``
- [ ] Build documentation locally to verify rendering: ``cd doc && make html``
- [ ] Check generated HTML for formatting issues
