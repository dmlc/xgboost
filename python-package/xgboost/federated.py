"""XGBoost Federated Learning related API."""

from .core import _LIB, XGBoostError, _check_call, build_info, c_str


def run_federated_server(
    port: int,
    world_size: int,
    server_key_path: str = "",
    server_cert_path: str = "",
    client_cert_path: str = "",
) -> None:
    """Run the Federated Learning server.

    Parameters
    ----------
    port : int
        The port to listen on.
    world_size: int
        The number of federated workers.
    server_key_path: str
        Path to the server private key file. SSL is turned off if empty.
    server_cert_path: str
        Path to the server certificate file. SSL is turned off if empty.
    client_cert_path: str
        Path to the client certificate file. SSL is turned off if empty.
    """
    if build_info()["USE_FEDERATED"]:
        if not server_key_path or not server_cert_path or not client_cert_path:
            _check_call(_LIB.XGBRunInsecureFederatedServer(port, world_size))
        else:
            _check_call(
                _LIB.XGBRunFederatedServer(
                    port,
                    world_size,
                    c_str(server_key_path),
                    c_str(server_cert_path),
                    c_str(client_cert_path),
                )
            )
    else:
        raise XGBoostError(
            "XGBoost needs to be built with the federated learning plugin "
            "enabled in order to use this module"
        )
