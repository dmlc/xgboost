from functools import wraps
from typing import Any, Callable

from ._typing import _T
from .config import config_context


def with_cuda_test(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Annotate a test to use CUDA."""

    @wraps(fn)
    def inner_fn(*args: Any, **kwargs: Any) -> _T:
        with config_context(device="CUDA:0"):
            return fn(*args, **kwargs)

    return inner_fn
