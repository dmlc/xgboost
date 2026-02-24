# API Documentation Review for Federated Learning

## Status: ✅ Mostly Complete, Minor Improvements Suggested

### Python API (`python-package/xgboost/federated.py`)

#### Current State: Good ✅
The Python API has proper NumPy-style docstrings:

**Classes:**
- ✅ `FederatedTracker` - Well documented with parameters

**Functions:**
- ⚠️ `run_federated_server()` - Needs enhancement

#### Suggested Improvement for `run_federated_server()`

**Current docstring is incomplete. Should be:**

```python
def run_federated_server(
    n_workers: int,
    port: int,
    server_key_path: Optional[str] = None,
    server_cert_path: Optional[str] = None,
    client_cert_path: Optional[str] = None,
    blocking: bool = True,
    timeout: int = 300,
) -> Optional[Dict[str, Any]]:
    """Run a federated learning server.

    This function starts a federated learning server that coordinates training
    across multiple workers without sharing raw data. Workers connect to this
    server to synchronize gradients and histograms in an encrypted manner.

    Parameters
    ----------
    n_workers : int
        The number of federated workers that will participate in training.
        All workers must connect before training begins.

    port : int
        The port number on which the federated server will listen.
        Workers must connect to this port.

    server_key_path : str, optional
        Path to the server's private key file (PEM format) for SSL/TLS encryption.
        Required when using secure mode. Default is None.

    server_cert_path : str, optional
        Path to the server's certificate file (PEM format) for SSL/TLS encryption.
        Required when using secure mode. Default is None.

    client_cert_path : str, optional
        Path to the client certificate file (PEM format) for SSL/TLS verification.
        Required when using secure mode. Default is None.

    blocking : bool, optional, default=True
        If True, blocks the calling thread until training completes.
        If False, launches the server in a background thread and returns worker
        arguments immediately. Use False when you need to programmatically
        manage worker connections.

    timeout : int, optional, default=300
        Maximum time in seconds to wait for all workers to connect.
        Raises an error if not all workers connect within this timeout.

    Returns
    -------
    worker_args : dict or None
        If blocking=False, returns a dictionary containing connection parameters
        for workers (server address, port, etc.). If blocking=True, returns None
        after training completes.

    Examples
    --------
    Secure federated server (blocking):

    >>> import xgboost.federated
    >>> xgboost.federated.run_federated_server(
    ...     n_workers=3,
    ...     port=9091,
    ...     server_key_path="server-key.pem",
    ...     server_cert_path="server-cert.pem",
    ...     client_cert_path="client-cert.pem",
    ... )

    Non-blocking server for custom worker management:

    >>> worker_args = xgboost.federated.run_federated_server(
    ...     n_workers=2,
    ...     port=9091,
    ...     blocking=False,
    ... )
    >>> # Start workers with worker_args...

    Notes
    -----
    * When using SSL/TLS (secure mode), all three certificate paths must be provided.
    * Generate self-signed certificates for testing:

      .. code-block:: bash

          openssl req -x509 -newkey rsa:2048 -days 7 -nodes \\
              -keyout server-key.pem -out server-cert.pem \\
              -subj "/C=US/CN=localhost"

    * For production, use properly signed certificates from a trusted CA.
    * The server must be started before any workers attempt to connect.

    See Also
    --------
    FederatedTracker : Lower-level tracker class for federated training.
    xgboost.collective.CommunicatorContext : Worker-side context manager.
    """
```

---

### C++ API (`plugin/federated/federated_plugin.h`)

#### Current State: Excellent ✅

The C++ header has comprehensive Doxygen documentation:
- ✅ All function prototypes documented
- ✅ Clear interface specifications
- ✅ Symbol names specified
- ✅ Return value documentation
- ✅ Parameter descriptions

**No changes needed** - this is already high quality.

---

### Missing Documentation

#### Testing Utilities (`python-package/xgboost/testing/federated.py`)

⚠️ **Needs Module-Level Docstring**

Add at the top of `python-package/xgboost/testing/federated.py`:

```python
"""Testing utilities for federated learning.

This module provides helper functions for testing federated learning
functionality in XGBoost. It includes utilities for:

- Starting federated servers and workers
- Generating SSL certificates for secure testing
- Running end-to-end federated training tests

These utilities are primarily intended for internal testing but can be
used as examples for setting up federated learning environments.

Examples
--------
Basic federated learning test setup:

>>> from xgboost.testing.federated import run_federated_learning
>>> # Run with 2 workers without SSL
>>> run_federated_learning(with_ssl=False, use_gpu=False, test_path=__file__)

Secure federated learning with SSL:

>>> run_federated_learning(with_ssl=True, use_gpu=False, test_path=__file__)
"""
```

**And enhance function docstrings:**

```python
def run_server(port: int, world_size: int, with_ssl: bool) -> None:
    """Run federated server for testing.

    Parameters
    ----------
    port : int
        Port number for the server to listen on.
    world_size : int
        Total number of workers expected to connect.
    with_ssl : bool
        Whether to enable SSL/TLS encryption.
    """

def run_worker(
    port: int, world_size: int, rank: int, with_ssl: bool, device: str
) -> None:
    """Run a federated client worker for testing.

    Parameters
    ----------
    port : int
        Port number to connect to on the server.
    world_size : int
        Total number of workers in the federated setup.
    rank : int
        Unique identifier for this worker (0 to world_size-1).
    with_ssl : bool
        Whether to use SSL/TLS encryption.
    device : str
        Device to use for training ('cpu' or 'cuda:N').
    """

def run_federated(world_size: int, with_ssl: bool, use_gpu: bool) -> None:
    """Launch federated server and workers for testing.

    Parameters
    ----------
    world_size : int
        Number of worker processes to create.
    with_ssl : bool
        Whether to enable SSL/TLS encryption.
    use_gpu : bool
        Whether to use GPU acceleration.
    """

def run_federated_learning(with_ssl: bool, use_gpu: bool, test_path: str) -> None:
    """Run complete federated learning test pipeline.

    This function:
    1. Generates SSL certificates (if with_ssl=True)
    2. Splits training/test data across workers
    3. Starts federated server
    4. Launches worker processes
    5. Runs distributed training
    6. Validates results

    Parameters
    ----------
    with_ssl : bool
        Whether to use SSL/TLS encryption for secure communication.
    use_gpu : bool
        Whether to use GPU acceleration for training.
    test_path : str
        Path to the test file, used to locate test data directory.
    """
```

---

## Summary

### ✅ Complete
- C++ plugin interface documentation
- `FederatedTracker` class documentation

### ⚠️ Needs Enhancement
- `run_federated_server()` - missing comprehensive docstring
- `testing/federated.py` - missing module docstring and function docstrings

### Action Items

1. **Update `python-package/xgboost/federated.py`:**
   - Replace `run_federated_server()` docstring with enhanced version above

2. **Update `python-package/xgboost/testing/federated.py`:**
   - Add module-level docstring
   - Add/enhance docstrings for all functions

---

## Verification

After updating, verify documentation builds correctly:

```bash
# Build documentation
cd doc
make html

# Check that federated API appears
# Look for: doc/_build/html/python/python_api.html
```

Ensure the federated module is exposed in the API docs by checking:
```python
# In python-package/xgboost/__init__.py
# Make sure federated is in __all__ or explicitly imported
```
