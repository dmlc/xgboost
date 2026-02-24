########################################
Security in Federated Learning
########################################

This document provides detailed information about the security features and architecture of XGBoost's federated learning implementation.

.. contents::
  :local:
  :backlinks: none

**********
Overview
**********

XGBoost's secure federated learning implementation protects data privacy during distributed training across multiple parties. The security model is based on research proposals documented in:

- `RFC #9987: Secure Vertical Federated Learning <https://github.com/dmlc/xgboost/issues/9987>`_
- `RFC #10170: Secure Horizontal Federated Learning <https://github.com/dmlc/xgboost/issues/10170>`_

**********
Threat Model
**********

Trust Assumptions
=================

The secure federated learning implementation assumes:

* **Trusted Partners**: A few trusted partners jointly train a model
* **Honest-but-Curious Parties**: Participants follow the protocol but may attempt to learn information from received data
* **Pre-Established Relationships**: Private Set Intersection (PSI) is completed before training
* **Network Security**: Fast, secure network connectivity between participants and central server
* **No Participant Dropout**: All parties remain available throughout training (dropout support is a non-goal)

Security Boundaries
===================

**What is Protected:**

* **Label Information**: Server/active party labels are protected from passive clients (vertical FL)
* **Feature Values**: Individual feature values not leaked between parties
* **Histograms**: Local gradient/Hessian histograms encrypted before transmission
* **Gradients**: Local gradients encrypted to prevent information leakage

**What is NOT Protected:**

* **Model Structure**: The final tree structure is shared among all parties
* **Aggregated Statistics**: Final aggregated histograms are revealed (in decrypted form)
* **Participation Patterns**: Which parties participate in training is observable

**********
Security Architecture
**********

SecureBoost Pattern (Vertical Federated Learning)
==================================================

The vertical federated learning implementation follows a variation of the `SecureBoost algorithm <https://arxiv.org/abs/1901.08755>`_.

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

The horizontal federated learning implementation uses homomorphic encryption to protect histogram data.

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

**Key Security Property:** Individual party histograms never leave their local environment in plaintext form.

**********
Plugin Architecture
**********

Processor Interface Pattern
============================

The security implementation uses a **processor interface** to decouple encryption from XGBoost core:

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

**Key Functions:**

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

**Architecture:**

* gRPC local handler performs encryption/decryption
* Uses external Python/C++ encryption libraries (e.g., ``tenseal``, ``python-paillier``)
* XGBoost communicates via processor interface
* No encryption dependencies in XGBoost binary

**Advantages:**

* Keeps XGBoost dependency-free
* Easy to update encryption libraries
* Multiple encryption schemes supported without rebuilding XGBoost
* Separate security audit scope

**Example with NVFlare:**

.. code-block:: python

    # NVFlare processor handles encryption
    from nvflare.app_common.xgb import FedXGBHistogramController

    controller = FedXGBHistogramController(
        num_rounds=10,
        # Encryption handled by NVFlare processor
        processor_class=XGBHomomorphicEncryptionProcessor
    )

Option 2: XGBoost-Side Encryption
----------------------------------

**Architecture:**

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

**********
Security Guarantees
**********

Vertical Federated Learning
============================

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
==============================

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

**********
Implementation Phases
**********

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

**********
Performance Considerations
**********

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

**********
Best Practices
**********

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

Privacy Risk Assessment
=======================

Before deploying secure federated learning:

1. **Threat Modeling**: Identify your specific adversaries and threats
2. **Data Sensitivity**: Assess sensitivity of labels, features, and aggregates
3. **Regulatory Compliance**: Ensure compliance with GDPR, HIPAA, etc.
4. **Differential Privacy**: Consider adding differential privacy for stronger guarantees
5. **Secure Computation Limitations**: Understand what is and isn't protected

**********
Integration with Frameworks
**********

NVFlare Integration
===================

NVIDIA FLARE (Federated Learning Application Runtime Environment) provides the recommended framework for secure federated learning:

.. code-block:: python

    # NVFlare server configuration
    from nvflare.app_common.xgb import FedXGBHistogramController

    controller = FedXGBHistogramController(
        num_rounds=100,
        early_stopping_rounds=10,
        # Encryption processor
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
        # Local data configuration
        world_size=5,
        rank=0
    )

See `NVFlare XGBoost Documentation <https://nvflare.readthedocs.io/en/main/examples/xgboost.html>`_ for complete examples.

**********
Testing Security Features
**********

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

**********
References
**********

Academic Papers
===============

* `SecureBoost: A Lossless Federated Learning Framework <https://arxiv.org/abs/1901.08755>`_ - Original SecureBoost algorithm
* `CKKS: Homomorphic Encryption for Arithmetic of Approximate Numbers <https://eprint.iacr.org/2016/421>`_ - CKKS scheme
* `Practical Secure Aggregation for Privacy-Preserving Machine Learning <https://eprint.iacr.org/2017/281>`_ - Secure aggregation

RFCs and Design Documents
==========================

* `RFC #9987: Secure Vertical Federated Learning <https://github.com/dmlc/xgboost/issues/9987>`_
* `RFC #10170: Secure Horizontal Federated Learning <https://github.com/dmlc/xgboost/issues/10170>`_

Libraries and Tools
===================

* `Microsoft SEAL <https://github.com/microsoft/SEAL>`_ - CKKS implementation
* `TenSEAL <https://github.com/OpenMined/TenSEAL>`_ - Python SEAL wrapper
* `python-paillier <https://github.com/data61/python-paillier>`_ - Paillier encryption
* `NVIDIA FLARE <https://github.com/NVIDIA/NVFlare>`_ - Federated learning framework

**********
FAQ
**********

**Q: Is the model trained with secure FL identical to centralized training?**

A: Yes, the final model is mathematically equivalent. The encryption only protects intermediate values (gradients, histograms) during transmission.

**Q: Can I use secure FL without a framework like NVFlare?**

A: Yes, but you'll need to implement the encryption plugin yourself following the interface in ``federated_plugin.h``.

**Q: What encryption scheme should I use?**

A: For horizontal FL, CKKS is recommended. For vertical FL, Paillier or CKKS both work. Consult a cryptography expert for production.

**Q: Is secure FL slower than regular FL?**

A: Yes, encryption adds overhead. Expect 2-10x slower depending on security parameters and hardware acceleration.

**Q: Does secure FL protect against malicious parties?**

A: No, the current design assumes honest-but-curious parties. Byzantine fault tolerance is a non-goal.

**Q: Can I add differential privacy on top of secure FL?**

A: Yes, differential privacy can be added independently to provide additional privacy guarantees.

For more questions, visit the `XGBoost Discussion Forum <https://discuss.xgboost.ai>`_.
