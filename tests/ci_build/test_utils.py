"""Utilities for the CI."""
import os
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, TypedDict, TypeVar, Union


class DirectoryExcursion:
    def __init__(self, path: Union[os.PathLike, str]) -> None:
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))

    def __enter__(self) -> None:
        os.chdir(self.path)

    def __exit__(self, *args: Any) -> None:
        os.chdir(self.curdir)


R = TypeVar("R")


def cd(path: Union[os.PathLike, str]) -> Callable:
    """Decorator for changing directory temporarily."""

    def chdir(func: Callable[..., R]) -> Callable[..., R]:
        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> R:
            with DirectoryExcursion(path):
                return func(*args, **kwargs)

        return inner

    return chdir


Record = TypedDict("Record", {"count": int, "total": timedelta})
timer: Dict[str, Record] = {}


def record_time(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator for recording function runtime."""
    global timer

    @wraps(func)
    def inner(*args: Any, **kwargs: Any) -> R:
        if func.__name__ not in timer:
            timer[func.__name__] = {"count": 0, "total": timedelta(0)}
        s = datetime.now()
        try:
            r = func(*args, **kwargs)
        finally:
            e = datetime.now()
            timer[func.__name__]["count"] += 1
            timer[func.__name__]["total"] += e - s
        return r

    return inner


def print_time() -> None:
    """Print all recorded items by :py:func:`record_time`."""
    global timer
    for k, v in timer.items():
        print(
            "Name:",
            k,
            "Called:",
            v["count"],
            "Elapsed:",
            f"{v['total'].seconds} secs",
        )


ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir
    )
)
R_PACKAGE = os.path.join(ROOT, "R-package")
JVM_PACKAGES = os.path.join(ROOT, "jvm-packages")
PY_PACKAGE = os.path.join(ROOT, "python-package")
