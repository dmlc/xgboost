"""XGBoost Experimental Federated Learning related API."""

import ctypes
from threading import Thread
from typing import Any, Dict, Optional

from .core import _LIB, _check_call, make_jcargs
from .tracker import RabitTracker


class FederatedTracker(RabitTracker):
    """Tracker for federated training.

    Parameters
    ----------
    n_workers :
        The number of federated workers.

    port :
        The port to listen on.

    secure :
        Whether this is a secure instance. If True, then the following arguments for SSL
        must be provided.

    server_key_path :
        Path to the server private key file.

    server_cert_path :
        Path to the server certificate file.

    client_cert_path :
        Path to the client certificate file.

    """

    def __init__(  # pylint: disable=R0913, W0231
        self,
        n_workers: int,
        port: int,
        secure: bool,
        server_key_path: Optional[str] = None,
        server_cert_path: Optional[str] = None,
        client_cert_path: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        handle = ctypes.c_void_p()
        args = make_jcargs(
            n_workers=n_workers,
            port=port,
            dmlc_communicator="federated",
            federated_secure=secure,
            server_key_path=server_key_path,
            server_cert_path=server_cert_path,
            client_cert_path=client_cert_path,
            timeout=int(timeout),
        )
        _check_call(_LIB.XGTrackerCreate(args, ctypes.byref(handle)))
        self.handle = handle


def run_federated_server(  # pylint: disable=too-many-arguments
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
    args: Dict[str, Any] = {"n_workers": n_workers}
    secure = all(
        path is not None
        for path in [server_key_path, server_cert_path, client_cert_path]
    )
    tracker = FederatedTracker(
        n_workers=n_workers,
        port=port,
        secure=secure,
        timeout=timeout,
        server_key_path=server_key_path,
        server_cert_path=server_cert_path,
        client_cert_path=client_cert_path,
    )
    tracker.start()

    if blocking:
        tracker.wait_for()
        return None

    thread = Thread(target=tracker.wait_for)
    thread.daemon = True
    thread.start()
    args.update(tracker.worker_args())
    return args
