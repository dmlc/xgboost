"""Collective module related utilities."""

import socket


def get_avail_port() -> int:
    """Returns a port that's available during the function call. It doesn't prevent the
    port from being used after the function returns as we can't reserve the port. The
    utility makes a test more likely to pass.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    return port
