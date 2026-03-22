"""Interpretability helpers for XGBoost models."""

from typing import Any, Optional, Tuple, Union

import numpy as np

from ._typing import ArrayLike, IterationRange
from .core import Booster, DMatrix, _deprecate_positional_args

Model = Union[Booster, Any]


def _get_booster(model: Model) -> Booster:
    if isinstance(model, Booster):
        return model

    get_booster = getattr(model, "get_booster", None)
    if not callable(get_booster):
        raise TypeError("`model` must be a Booster or expose `get_booster()`.")

    booster = get_booster()
    if not isinstance(booster, Booster):
        raise TypeError("`model.get_booster()` must return an xgboost.Booster.")
    return booster


def _get_iteration_range(
    model: Model, iteration_range: Optional[IterationRange]
) -> IterationRange:
    get_iteration_range = getattr(model, "_get_iteration_range", None)
    if callable(get_iteration_range):
        return get_iteration_range(iteration_range)

    if iteration_range is None:
        return (0, 0)
    return iteration_range


def _to_dmatrix(model: Model, data: Union[DMatrix, ArrayLike]) -> DMatrix:
    if isinstance(data, DMatrix):
        return data

    return DMatrix(
        data,
        missing=getattr(model, "missing", np.nan),
        nthread=getattr(model, "n_jobs", None),
        feature_types=getattr(model, "feature_types", None),
        enable_categorical=getattr(model, "enable_categorical", True),
    )


@_deprecate_positional_args
def shap_values(  # pylint: disable=too-many-arguments
    model: Model,
    data: Union[DMatrix, ArrayLike],
    *,
    iteration_range: Optional[IterationRange] = None,
    approx_contribs: bool = False,
    validate_features: bool = True,
    strict_shape: bool = False,
    return_bias: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute SHAP values from an XGBoost model.

    The returned values exclude the bias contribution by default. Set
    ``return_bias=True`` to return ``(values, bias)``.

    Parameters
    ----------
    model :
        A :py:class:`~xgboost.core.Booster` or sklearn-style XGBoost model.
    data :
        Input data. Array-like input is accepted and converted to
        :py:class:`~xgboost.core.DMatrix`.
    iteration_range :
        Layer range used for prediction.
    approx_contribs :
        Whether to use approximation for SHAP contributions.
    validate_features :
        Whether to validate feature names.
    strict_shape :
        Whether to keep output shape invariant.
    return_bias :
        If True, return ``(values, bias)``. If False, return only ``values``.
    """
    booster = _get_booster(model)
    matrix = _to_dmatrix(model, data)
    iter_range = _get_iteration_range(model, iteration_range)

    contribs = booster.predict(
        matrix,
        pred_contribs=True,
        approx_contribs=approx_contribs,
        validate_features=validate_features,
        iteration_range=iter_range,
        strict_shape=strict_shape,
    )
    values = contribs[..., :-1]

    if return_bias:
        return values, contribs[..., -1]
    return values


@_deprecate_positional_args
def shap_interactions(  # pylint: disable=too-many-arguments
    model: Model,
    data: Union[DMatrix, ArrayLike],
    *,
    iteration_range: Optional[IterationRange] = None,
    approx_contribs: bool = False,
    validate_features: bool = True,
    strict_shape: bool = False,
) -> np.ndarray:
    """Compute SHAP interaction values from an XGBoost model.

    The returned tensor excludes the bias row and column.
    """
    booster = _get_booster(model)
    matrix = _to_dmatrix(model, data)
    iter_range = _get_iteration_range(model, iteration_range)

    interactions = booster.predict(
        matrix,
        pred_interactions=True,
        approx_contribs=approx_contribs,
        validate_features=validate_features,
        iteration_range=iter_range,
        strict_shape=strict_shape,
    )
    return interactions[..., :-1, :-1]


__all__ = ["shap_values", "shap_interactions"]
