"""XGBoost Federated Learning related API."""

from .core import _LIB, _check_call, c_str


def run_federated_server(port: int,
                         world_size: int,
                         server_key_path: str,
                         server_cert_path: str,
                         client_cert_path: str) -> None:
    """Run the Federated Learning server.

    Parameters
    ----------
    port : int
        The port to listen on.
    world_size: int
        The number of federated workers.
    server_key_path: str
        Path to the server private key file.
    server_cert_path: str
        Path to the server certificate file.
    client_cert_path: str
        Path to the client certificate file.
    """
    _check_call(_LIB.XGBRunFederatedServer(port,
                                           world_size,
                                           c_str(server_key_path),
                                           c_str(server_cert_path),
                                           c_str(client_cert_path)))
