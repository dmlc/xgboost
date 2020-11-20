"""Global configuration for XGBoost"""
import ctypes
from contextlib import contextmanager

from .core import _LIB, _check_call


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
    str_array_t = ctypes.c_char_p * len(new_config)
    names, values = str_array_t(), str_array_t()
    for i, (key, value) in enumerate(new_config.items()):
        names[i] = key.encode('utf-8')
        values[i] = str(value).encode('utf-8')

    _check_call(_LIB.XGBSetGlobalConfig(names, values, ctypes.c_size_t(len(new_config))))


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
    str_array_t = ctypes.POINTER(ctypes.c_char_p)
    name = str_array_t()
    value = str_array_t()
    num_param = ctypes.c_size_t()
    _check_call(_LIB.XGBGetGlobalConfig(
        ctypes.byref(name),
        ctypes.byref(value),
        ctypes.byref(num_param)))
    num_param = num_param.value
    assert num_param > 0
    params = {}
    for i in range(num_param):
        params[name[i].decode('utf-8')] = value[i].decode('utf-8')

    return params


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
