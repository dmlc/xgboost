# pylint: disable=missing-function-docstring
"""Global configuration for XGBoost"""
import ctypes
import json
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, cast

from ._typing import _F
from .core import _LIB, _check_call, c_str, py_str


def config_doc(
    *,
    header: Optional[str] = None,
    extra_note: Optional[str] = None,
    parameters: Optional[str] = None,
    returns: Optional[str] = None,
    see_also: Optional[str] = None,
) -> Callable[[_F], _F]:
    """Decorator to format docstring for config functions.

    Parameters
    ----------
    header: str
        An introducion to the function
    extra_note: str
        Additional notes
    parameters: str
        Parameters of the function
    returns: str
        Return value
    see_also: str
        Related functions
    """

    doc_template = """
    {header}

    Global configuration consists of a collection of parameters that can be applied in the
    global scope. See :ref:`global_config` for the full list of parameters supported in
    the global configuration.

    {extra_note}

    .. versionadded:: 1.4.0
    """

    common_example = """
    Example
    -------

    .. code-block:: python

        import xgboost as xgb

        # Show all messages, including ones pertaining to debugging
        xgb.set_config(verbosity=2)

        # Get current value of global configuration
        # This is a dict containing all parameters in the global configuration,
        # including 'verbosity'
        config = xgb.get_config()
        assert config['verbosity'] == 2

        # Example of using the context manager xgb.config_context().
        # The context manager will restore the previous value of the global
        # configuration upon exiting.
        with xgb.config_context(verbosity=0):
            # Suppress warning caused by model generated with XGBoost version < 1.0.0
            bst = xgb.Booster(model_file='./old_model.bin')
        assert xgb.get_config()['verbosity'] == 2  # old value restored

    Nested configuration context is also supported:

    Example
    -------

    .. code-block:: python

        with xgb.config_context(verbosity=3):
            assert xgb.get_config()["verbosity"] == 3
            with xgb.config_context(verbosity=2):
                assert xgb.get_config()["verbosity"] == 2

        xgb.set_config(verbosity=2)
        assert xgb.get_config()["verbosity"] == 2
        with xgb.config_context(verbosity=3):
            assert xgb.get_config()["verbosity"] == 3
    """

    def none_to_str(value: Optional[str]) -> str:
        return "" if value is None else value

    def config_doc_decorator(func: _F) -> _F:
        func.__doc__ = (
            doc_template.format(
                header=none_to_str(header), extra_note=none_to_str(extra_note)
            )
            + none_to_str(parameters)
            + none_to_str(returns)
            + none_to_str(common_example)
            + none_to_str(see_also)
        )

        @wraps(func)
        def wrap(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(_F, wrap)

    return config_doc_decorator


@config_doc(
    header="""
    Set global configuration.
    """,
    parameters="""
    Parameters
    ----------
    new_config: Dict[str, Any]
        Keyword arguments representing the parameters and their values
            """,
)
def set_config(**new_config: Any) -> None:
    not_none = {}
    for k, v in new_config.items():
        if v is not None:
            not_none[k] = v
    config = json.dumps(not_none)
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))


@config_doc(
    header="""
    Get current values of the global configuration.
    """,
    returns="""
    Returns
    -------
    args: Dict[str, Any]
        The list of global parameters and their values
            """,
)
def get_config() -> Dict[str, Any]:
    config_str = ctypes.c_char_p()
    _check_call(_LIB.XGBGetGlobalConfig(ctypes.byref(config_str)))
    value = config_str.value
    assert value
    config = json.loads(py_str(value))
    return config


@contextmanager
@config_doc(
    header="""
    Context manager for global XGBoost configuration.
    """,
    parameters="""
    Parameters
    ----------
    new_config: Dict[str, Any]
        Keyword arguments representing the parameters and their values
            """,
    extra_note="""
    .. note::

        All settings, not just those presently modified, will be returned to their
        previous values when the context manager is exited. This is not thread-safe.
            """,
    see_also="""
    See Also
    --------
    set_config: Set global XGBoost configuration
    get_config: Get current values of the global configuration
            """,
)
def config_context(**new_config: Any) -> Iterator[None]:
    old_config = get_config().copy()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
