# Parameter Documentation for Federated Learning

## Location
Add to: `doc/parameter.rst`

## Section to Add

Insert this section after the "Learning Task Parameters" section in `doc/parameter.rst`:

---

```rst
****************************
Federated Learning Parameters
****************************

These parameters configure federated learning, which enables training across multiple parties without sharing raw data. All federated parameters are set through the :py:class:`xgboost.collective.CommunicatorContext`.

.. note:: Federated learning requires ``tree_method='hist'``

  Only the histogram-based tree method (``hist``) supports federated learning. Other tree methods (``exact``, ``approx``) are not compatible with federated mode.

Communicator Selection
=======================

* ``dmlc_communicator`` [default= ``rabit``]

  - Specifies the communication backend for distributed training. For federated learning, this must be set to ``federated``.
  - Valid values:

    + ``rabit``: Standard distributed training (MPI-like)
    + ``federated``: Federated learning mode

  - Example:

    .. code-block:: python

        with xgb.collective.CommunicatorContext(dmlc_communicator='federated'):
            # training code

Required Federated Parameters
==============================

When ``dmlc_communicator='federated'``, the following parameters are required:

* ``federated_server_address``

  - Address of the federated server in the format ``hostname:port`` or ``ip:port``.
  - The server must be started before workers attempt to connect.
  - Example: ``'localhost:9091'`` or ``'10.0.1.5:9091'``

* ``federated_world_size``

  - Total number of workers (parties) participating in federated training.
  - Must be a positive integer.
  - All workers must connect before training begins.
  - Example: ``2`` (for two parties)

* ``federated_rank``

  - Unique identifier for this worker, starting from 0.
  - Must be in range ``[0, world_size - 1]``.
  - Each worker must have a different rank.
  - Example: ``0`` for first worker, ``1`` for second worker, etc.

SSL/TLS Parameters (Optional but Recommended)
==============================================

For secure federated learning with encrypted communication:

* ``federated_server_cert_path``

  - Path to the server certificate file (PEM format).
  - Required on worker side when using SSL/TLS.
  - Used to verify the server's identity.
  - Example: ``'/path/to/server-cert.pem'``

* ``federated_client_key_path``

  - Path to the client private key file (PEM format).
  - Required on worker side when using SSL/TLS.
  - Used for client authentication.
  - Example: ``'/path/to/client-key.pem'``

* ``federated_client_cert_path``

  - Path to the client certificate file (PEM format).
  - Required on worker side when using SSL/TLS.
  - Used for client authentication.
  - Example: ``'/path/to/client-cert.pem'``

.. warning:: Production Deployments

  Always use SSL/TLS in production environments to prevent eavesdropping and man-in-the-middle attacks. All three certificate parameters must be provided together.

Advanced Federated Parameters
==============================

* ``federated_plugin``

  - Configuration for third-party encryption plugins (advanced users).
  - Allows custom secure aggregation protocols.
  - Value should be a JSON object with plugin-specific settings.
  - See ``plugin/federated/federated_plugin.h`` for plugin interface details.
  - Example:

    .. code-block:: python

        communicator_env = {
            'dmlc_communicator': 'federated',
            # ... other params ...
            'federated_plugin': {
                'path': '/path/to/plugin.so',
                'config': {'encryption': 'homomorphic'}
            }
        }

Complete Example
================

Secure federated worker configuration:

.. code-block:: python

    import xgboost as xgb

    communicator_env = {
        # Required: Select federated communicator
        'dmlc_communicator': 'federated',

        # Required: Server connection
        'federated_server_address': 'federated-server.example.com:9091',
        'federated_world_size': 3,
        'federated_rank': 0,  # This is worker 0

        # Optional but recommended: SSL/TLS
        'federated_server_cert_path': '/etc/xgboost/certs/server-cert.pem',
        'federated_client_key_path': '/etc/xgboost/certs/client-key.pem',
        'federated_client_cert_path': '/etc/xgboost/certs/client-cert.pem',
    }

    with xgb.collective.CommunicatorContext(**communicator_env):
        dtrain = xgb.DMatrix('local_data.txt')

        params = {
            'max_depth': 5,
            'eta': 0.3,
            'objective': 'binary:logistic',
            'tree_method': 'hist',  # Required for federated learning
            'device': 'cuda:0',     # Optional: GPU acceleration
        }

        bst = xgb.train(params, dtrain, num_boost_round=100)

        # Save model only on rank 0
        if xgb.collective.get_rank() == 0:
            bst.save_model('federated_model.json')

Server-Side Parameters
======================

Server parameters are configured via the :py:func:`xgboost.federated.run_federated_server` function:

* ``n_workers``

  - Number of workers expected to connect.
  - Must match ``federated_world_size`` on the worker side.

* ``port``

  - Port number for the server to listen on.
  - Must be accessible to all workers.

* ``server_key_path``, ``server_cert_path``, ``client_cert_path``

  - SSL/TLS certificate paths for secure mode.
  - All three must be provided together for secure mode.

* ``timeout``

  - Maximum seconds to wait for all workers to connect.
  - Default: 300 seconds (5 minutes).

* ``blocking``

  - If ``True``, blocks until training completes.
  - If ``False``, runs in background thread.
  - Default: ``True``.

See :doc:`/tutorials/federated_learning` for detailed usage examples.

Compatibility Notes
===================

* Federated learning is available starting from XGBoost 2.1.0
* Only the ``hist`` tree method is supported
* GPU acceleration (``device='cuda'``) is supported
* All workers must use the same XGBoost version
* Network latency affects training speed
```

---

## Verification

After adding this section, build the documentation and verify:

```bash
cd doc
make html

# Open in browser
open _build/html/parameter.html

# Check that the "Federated Learning Parameters" section appears
```

## Integration with Existing doc/parameter.rst

Insert the above section **after** the "Learning Task Parameters" section and **before** the "Command Line Parameters" section.

The structure should be:

```
1. Global Configuration
2. General Parameters
3. Parameters for Tree Booster
4. Parameters for Linear Booster
5. Learning Task Parameters
6. **Federated Learning Parameters** <-- ADD HERE
7. Command Line Parameters
```

## Cross-References to Add

In other relevant sections, add cross-references to federated learning:

### In "tree_method" parameter description:

```rst
* ``tree_method`` string [default= ``auto``]

  - The tree construction algorithm used in XGBoost. See description in the :doc:`/treemethod` for more details.
  - XGBoost supports ``hist``, ``approx``, and ``exact``.
  - **Note:** For federated learning, only ``hist`` is supported. See :ref:`federated_parameters`.
```

### In "device" parameter description:

```rst
* ``device`` [default= ``cpu``]

  ...existing description...

  For federated learning with GPU support, see :doc:`/tutorials/federated_learning`.
```
