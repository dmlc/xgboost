"""Global configuration for XGBoost"""
import ctypes
import json
from contextlib import contextmanager

from .core import _LIB, _check_call, c_str, py_str


def set_config(**new_config):
    """Set global configuration, collection of parameters that apply globally. See
    https://xgboost.readthedocs.io/en/latest/parameter.html for the full list of parameters
    supported in the global configuration.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    new_config: Dict[str, str]
        Keyword arguments representing the parameters and their values

    Example
    -------

    .. code-block:: python

        import xgboost as xgb

        # Silence all messages
        xgb.set_config(verbosity=0)
    """
    config_str = json.dumps(new_config)

    _check_call(_LIB.XGBSetGlobalConfig(c_str(config_str)))


def get_config():
    """
    Get current values of the global configuration.
    See https://xgboost.readthedocs.io/en/latest/parameter.html for the full list of parameters
    supported in the global configuration.

    .. versionadded:: 1.3.0

    Returns
    -------
    args: Dict[str, str]
        The list of global parameters and their values
    """
    config_str = ctypes.c_char_p()
    _check_call(_LIB.XGBGetGlobalConfig(ctypes.byref(config_str)))
    config = json.loads(py_str(config_str.value))

    return config


@contextmanager
def config_context(**new_config):
    """
    Context manager for global XGBoost configuration. Global configuration consists of a collection
    of parameters that apply globally. See https://xgboost.readthedocs.io/en/latest/parameter.html
    for the full list of parameters supported in the global configuration.

    .. note::

        All settings, not just those presently modified, will be returned to their previous values
        when the context manager is exited. This is not thread-safe.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    new_config: Dict[str, str]
        Keyword arguments representing the parameters and their values

    Example
    -------

    .. code-block:: python

        import xgboost as xgb

        # Suppress warning caused by model generated with XGBoost version < 1.0.0
        with xgb.config_context(verbosity=0):
            bst = xgb.Booster(model_file='./old_model.bin')

    See Also
    --------
    set_config: Set global XGBoost configuration
    get_config: Get current values of the global configuration
    """
    old_config = get_config().copy()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
