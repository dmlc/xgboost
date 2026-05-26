"""Interpretability functions for XGBoost models."""

from typing import Optional, Tuple, Union

import numpy as np

from ._typing import ArrayLike, FloatCompatible, IterationRange
from .core import Booster, DMatrix


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


def _predict_contribs(
    booster: Booster,
    data: DMatrix,
    *,
    device: Optional[str],
    **kwargs: object,
) -> np.ndarray:
    if device is None:
        return booster.predict(data, **kwargs)

    config = booster.save_config()
    try:
        booster.set_param({"device": device})
        return booster.predict(data, **kwargs)
    finally:
        booster.load_config(config)


def shap_values(  # pylint: disable=too-many-arguments
    model: object,
    X: Union[DMatrix, ArrayLike],
    *,
    X_background: Optional[Union[DMatrix, ArrayLike]] = None,
    device: Optional[str] = None,
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
    device :
        Optional prediction device override, such as ``"cpu"``, ``"cuda"``, or
        ``"cuda:0"``. The model's original configuration is restored after
        prediction. This option temporarily mutates the underlying Booster and
        is not safe for concurrent use of the same model.
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
    """
    if X_background is not None:
        raise NotImplementedError("`X_background` is not yet supported.")
    # Existing SHAP prediction always returns margin contributions. Keep this
    # argument in the initial API so callers can use the proposed signature.
    _ = output_margin

    booster = _as_booster(model)
    data = _as_prediction_dmatrix(model, X, missing)
    contribs = _predict_contribs(
        booster,
        data,
        device=device,
        pred_contribs=True,
        validate_features=validate_features,
        iteration_range=_get_iteration_range(model, iteration_range),
    )

    values = contribs[..., :-1]
    bias = contribs[..., -1]
    return values, bias


__all__ = ["shap_values"]
