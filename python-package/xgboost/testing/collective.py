"""Collective modele related utilities."""

import socket


def get_avail_port() -> int:
    """Return a port that's available during the function call. It doesn't prevent the
    port being used after the function returns. We can't reserve a port. The utility
    makes a test more likely to pass.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    return port
