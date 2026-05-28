"""Interpretability functions for XGBoost models."""

import ctypes
import json
from typing import Optional, Tuple, Union

import numpy as np

from ._typing import ArrayLike, FloatCompatible, IterationRange
from .core import (
    _LIB,
    Booster,
    DMatrix,
    _check_call,
    _prediction_output,
    c_bst_ulong,
    from_pystr_to_cstr,
)


def _as_booster(model: object) -> Booster:
    if isinstance(model, Booster):
        return model
    get_booster = getattr(model, "get_booster", None)
    if not callable(get_booster):
        raise TypeError(
            "`model` must be an xgboost.Booster or an object with get_booster()."
        )
    booster = get_booster()
    if not isinstance(booster, Booster):
        raise TypeError("`model.get_booster()` must return an xgboost.Booster.")
    return booster


def _get_iteration_range(
    model: object, iteration_range: Optional[IterationRange]
) -> IterationRange:
    get_iteration_range = getattr(model, "_get_iteration_range", None)
    if get_iteration_range is not None:
        return get_iteration_range(iteration_range)
    if iteration_range is None:
        return (0, 0)
    return iteration_range


def _as_prediction_dmatrix(
    model: object, X: Union[DMatrix, ArrayLike], missing: Optional[FloatCompatible]
) -> DMatrix:
    if isinstance(X, DMatrix):
        if missing is not None:
            raise ValueError("`missing` must not be specified when `X` is a DMatrix.")
        return X

    return DMatrix(
        X,
        missing=missing if missing is not None else getattr(model, "missing", None),
        nthread=getattr(model, "n_jobs", None),
        feature_types=getattr(model, "feature_types", None),
        enable_categorical=getattr(model, "enable_categorical", False),
    )


def _capi_shap_values(
    booster: Booster,
    data: DMatrix,
    background: Optional[DMatrix],
    iteration_range: IterationRange,
) -> Tuple[np.ndarray, np.ndarray]:
    values_shape = ctypes.POINTER(c_bst_ulong)()
    values_dim = c_bst_ulong()
    values = ctypes.POINTER(ctypes.c_float)()
    bias_shape = ctypes.POINTER(c_bst_ulong)()
    bias_dim = c_bst_ulong()
    bias = ctypes.POINTER(ctypes.c_float)()
    config = {
        "algorithm": "auto",
        "iteration_begin": int(iteration_range[0]),
        "iteration_end": int(iteration_range[1]),
    }
    _check_call(
        _LIB.XGBoosterInterpretShapValues(
            booster.handle,
            data.handle,
            background.handle if background is not None else None,
            from_pystr_to_cstr(json.dumps(config)),
            ctypes.byref(values_shape),
            ctypes.byref(values_dim),
            ctypes.byref(values),
            ctypes.byref(bias_shape),
            ctypes.byref(bias_dim),
            ctypes.byref(bias),
        )
    )
    values_out = _prediction_output(values_shape, values_dim, values, False)
    bias_out = _prediction_output(bias_shape, bias_dim, bias, False)
    if values_out.shape[-1] == 1:
        values_out = values_out[..., 0]
        bias_out = bias_out[..., 0]
    return values_out, bias_out


def shap_values(  # pylint: disable=too-many-arguments
    model: object,
    X: Union[DMatrix, ArrayLike],
    *,
    X_background: Optional[Union[DMatrix, ArrayLike]] = None,
    output_margin: bool = False,
    iteration_range: Optional[IterationRange] = None,
    missing: Optional[FloatCompatible] = None,
    validate_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return SHAP values for an XGBoost model.

    This function accepts either a :py:class:`xgboost.Booster` or an sklearn-style
    XGBoost model and returns feature contributions together with the separated
    bias term.

    Parameters
    ----------
    model :
        XGBoost booster or sklearn-style XGBoost model.
    X :
        Input data.
    X_background :
        Background data for interventional SHAP values. This is reserved for a
        future implementation and is currently unsupported.
    output_margin :
        Accepted for API compatibility. SHAP contributions currently correspond
        to the model margin.
    iteration_range :
        Specifies which layer of trees are used in prediction.
    missing :
        Value in array-like ``X`` to treat as missing. When None, use the
        model's missing value if available, otherwise ``np.nan``. This must not
        be specified when ``X`` is already a DMatrix.
    validate_features :
        Validate feature names between the model and input data.

    Returns
    -------
    values, bias :
        ``values`` contains feature SHAP values with the bias term removed.
        ``bias`` contains the separated bias term. For multi-target models, the
        output shape follows the corresponding prediction shape with the final
        feature dimension split into ``values`` and ``bias``.

    Notes
    -----
    To use GPU algorithms, configure the model before calling this function, for
    example with ``booster.set_param({"device": "cuda"})``.
    """
    # SHAP contributions currently correspond to the model margin. Keep this
    # argument in the initial API so callers can use the proposed signature.
    _ = output_margin

    booster = _as_booster(model)
    data = _as_prediction_dmatrix(model, X, missing)
    if validate_features:
        validate = getattr(booster, "_validate_features")
        validate(data.feature_names)
    background = (
        _as_prediction_dmatrix(model, X_background, missing=None)
        if X_background is not None
        else None
    )
    return _capi_shap_values(
        booster,
        data,
        background,
        _get_iteration_range(model, iteration_range),
    )


__all__ = ["shap_values"]
