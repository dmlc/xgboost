# pylint: disable=missing-function-docstring
"""Global configuration for XGBoost"""
import ctypes
import json
from contextlib import contextmanager
from functools import wraps

from .core import _LIB, _check_call, c_str, py_str


def config_doc(*, header=None, extra_note=None, parameters=None, returns=None,
               see_also=None):
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
    """

    def none_to_str(value):
        return '' if value is None else value

    def config_doc_decorator(func):
        func.__doc__ = (doc_template.format(header=none_to_str(header),
                                            extra_note=none_to_str(extra_note))
                        + none_to_str(parameters) + none_to_str(returns)
                        + none_to_str(common_example) + none_to_str(see_also))

        @wraps(func)
        def wrap(*args, **kwargs):
            return func(*args, **kwargs)
        return wrap
    return config_doc_decorator


@config_doc(header="""
    Set global configuration.
    """,
            parameters="""
    Parameters
    ----------
    new_config: Dict[str, Any]
        Keyword arguments representing the parameters and their values
            """)
def set_config(**new_config):
    config = json.dumps(new_config)
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))


@config_doc(header="""
    Get current values of the global configuration.
    """,
            returns="""
    Returns
    -------
    args: Dict[str, Any]
        The list of global parameters and their values
            """)
def get_config():
    config_str = ctypes.c_char_p()
    _check_call(_LIB.XGBGetGlobalConfig(ctypes.byref(config_str)))
    config = json.loads(py_str(config_str.value))
    return config


@contextmanager
@config_doc(header="""
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
            """)
def config_context(**new_config):
    old_config = get_config().copy()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
