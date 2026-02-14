"""
A demo for multi-output regression using reduced gradient
=========================================================

See :doc:`/tutorials/multioutput` for more information.

.. versionadded:: 3.2.0

.. note::

    The implementation is experimental and many features are missing.

.. seealso:: :ref:`sphx_glr_python_examples_multioutput_regression.py`

"""

import argparse
from typing import Protocol, Tuple, cast

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression

import xgboost as xgb
from xgboost.objective import TreeObjective


class SupportsGet(Protocol):
    """Array-like object exposing a NumPy conversion via `get`."""

    def get(self) -> np.ndarray: ...


class LsObjMean(TreeObjective):
    """Least squared error. Reduce the size of the gradient using mean value."""

    def __init__(self, device: str) -> None:
        self.device = device

    def __call__(
        self, iteration: int, y_pred: np.ndarray, dtrain: xgb.DMatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_true = dtrain.get_label()
        grad = y_pred - y_true
        if self.device == "cpu":
            hess = np.ones(grad.shape)
            return grad, hess
        else:
            import cupy as cp

            hess = cp.ones(grad.shape)

            return cp.array(grad), cp.array(hess)

    def split_grad(
        self, iteration: int, grad: np.ndarray, hess: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.device == "cpu":
            sgrad = np.mean(grad, axis=1)
            shess = np.mean(hess, axis=1)
        else:
            import cupy as cp

            sgrad = cp.mean(grad, axis=1)
            shess = cp.mean(hess, axis=1)
        return sgrad, shess


def svd_class(device: str) -> BaseEstimator:
    """One of the methods in the sketch boost paper."""
    from sklearn.decomposition import TruncatedSVD

    svd_params = {"algorithm": "arpack", "n_components": 2, "n_iter": 8}
    svd = TruncatedSVD(**svd_params)
    return svd


class LsObjSvd(LsObjMean):
    """Reduce the size of the gradient using SVD."""

    def __init__(self, device: str) -> None:
        super().__init__(device=device)

    def split_grad(
        self, iteration: int, grad: np.ndarray, hess: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        svd = svd_class(self.device)
        if self.device == "cuda":
            grad = cast(SupportsGet, grad).get()
            hess = cast(SupportsGet, hess).get()

        svd.fit(grad)
        grad = svd.transform(grad)
        hess = svd.transform(hess)
        if self.device == "cpu":
            hess = np.clip(hess, 0.01, None)
        else:
            import cupy as cp

            hess = cp.clip(hess, 0.01, None)
        return grad, hess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    X, y = make_regression(
        n_samples=8192, n_features=32, n_targets=8, random_state=2026
    )
    Xy = xgb.QuantileDMatrix(X, y)

    for obj in (LsObjMean(args.device), LsObjSvd(args.device)):
        xgb.train(
            {
                "device": args.device,
                "multi_strategy": "multi_output_tree",
            },
            Xy,
            evals=[(Xy, "Train")],
            obj=obj,
            num_boost_round=16,
        )


if __name__ == "__main__":
    main()
