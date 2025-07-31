"""Helpers for test code."""

from typing import Any, Literal, TypeAlias

import numpy as np

from ..compat import import_cupy
from ..data import _is_cupy_alike

Device: TypeAlias = Literal["cpu", "cuda"]


def assert_allclose(
    device: Device, a: Any, b: Any, *, rtol: float = 1e-7, atol: float = 0
) -> None:
    """Dispatch the assert_allclose for devices."""
    if device == "cpu" and not _is_cupy_alike(a) and not _is_cupy_alike(b):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    else:
        cp = import_cupy()
        cp.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
