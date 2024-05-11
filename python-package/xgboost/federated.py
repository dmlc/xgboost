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
        server_key_path: str = "",
        server_cert_path: str = "",
        client_cert_path: str = "",
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
    timeout: int = 300,
) -> Dict[str, Any]:
    """See :py:class:`~xgboost.federated.FederatedTracker` for more info."""
    args: Dict[str, Any] = {"n_workers": n_workers}
    secure = all(
        path is not None
        for path in [server_key_path, server_cert_path, client_cert_path]
    )
    tracker = FederatedTracker(
        n_workers=n_workers, port=port, secure=secure, timeout=timeout
    )
    tracker.start()

    thread = Thread(target=tracker.wait_for)
    thread.daemon = True
    thread.start()
    args.update(tracker.worker_args())
    return args
