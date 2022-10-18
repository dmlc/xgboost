"""Utilities for defining Python tests."""
# pylint: disable=unused-import
import os
import socket
from platform import system
from typing import Any, TypedDict

CURDIR = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
PROJECT_ROOT = os.path.normpath(os.path.join(CURDIR, os.path.pardir, os.path.pardir))


PytestSkip = TypedDict("PytestSkip", {"condition": bool, "reason": str})


def has_ipv6() -> bool:
    """Check whether IPv6 is enabled on this host."""
    # connection error in macos, still need some fixes.
    if system() not in ("Linux", "Windows"):
        return False

    if socket.has_ipv6:
        try:
            with socket.socket(
                socket.AF_INET6, socket.SOCK_STREAM
            ) as server, socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as client:
                server.bind(("::1", 0))
                port = server.getsockname()[1]
                server.listen()

                client.connect(("::1", port))
                conn, _ = server.accept()

                client.sendall("abc".encode())
                msg = conn.recv(3).decode()
                # if the code can be executed to this point, the message should be
                # correct.
                assert msg == "abc"
            return True
        except OSError:
            pass
    return False


def skip_ipv6() -> PytestSkip:
    """PyTest skip mark for IPv6."""
    return {"condition": not has_ipv6(), "reason": "IPv6 is required to be enabled."}


def skip_spark() -> PytestSkip:
    """Pytest skip mark for PySpark tests."""
    if system() != "Linux":
        return {"condition": True, "reason": "Unsupported platform."}

    try:
        import pyspark  # noqa

        # just in case there's a pyspark stub created by some other libraries
        from pyspark.ml import Pipeline  # noqa

        spark_installed = True
    except ImportError:
        spark_installed = False
    return {"condition": not spark_installed, "reason": "Spark is not installed"}


def timeout(sec: int, *args: Any, enable: bool = True, **kwargs: Any) -> Any:
    """Make a pytest mark for the `pytest-timeout` package.

    Parameters
    ----------
    sec :
        Timeout seconds.
    enable :
        Control whether timeout should be applied, used for debugging.

    Returns
    -------
    pytest.mark.timeout
    """
    import pytest  # pylint: disable=import-error

    # This is disabled for now due to regression caused by conflicts between federated
    # learning build and the CI container environment.
    if enable:
        return pytest.mark.timeout(sec, *args, **kwargs)
    return pytest.mark.timeout(None, *args, **kwargs)
