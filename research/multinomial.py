"""
Shared primitives for the exact-multinomial research scripts.

This module deliberately contains only the pieces that must NOT differ between
``exact_multiclass_hessian.py`` and ``reproduce_issue_12278.py``:

  * a numerically stable softmax,
  * the reference-class (K-1 free logit) probability map,
  * a finite-difference helper with a principled step size,
  * one Newton driver with a single, explicit stopping criterion.

Everything that is genuinely under investigation -- the objective expression, the
gradient, and the Hessian (or its approximation) -- is defined separately in each
script. That separation is intentional: ``reproduce_issue_12278.py`` must be free
to reproduce the upstream objective expression faithfully, including its known
inconsistency, without inheriting this module's opinion of what the objective
"should" be.

The Newton driver judges every method by the same exact-gradient criterion, so a
convergence comparison cannot be biased by giving one method an easier test.

No XGBoost import. These scripts are an independent check on the mathematics.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "softmax",
    "reference_probabilities",
    "central_difference_step",
    "finite_difference_jacobian",
    "newton_descent",
    "NewtonResult",
]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the row maximum leaves the result unchanged mathematically
    (softmax is invariant to a constant shift) but bounds the exponent at zero,
    so ``exp`` cannot overflow. The denominator is then at least 1, so it cannot
    underflow to zero either.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def reference_probabilities(free_logits: np.ndarray) -> np.ndarray:
    """Probabilities for the K-1 free-logit parameterization.

    ``free_logits`` holds z_0 .. z_{K-2}. The last class is the reference class
    and its logit is pinned at exactly zero, which fixes the gauge freedom of the
    softmax. See the module docstring of ``exact_multiclass_hessian.py`` for why
    this is necessary.
    """
    free_logits = np.asarray(free_logits, dtype=np.float64)
    if free_logits.ndim != 1:
        raise ValueError("free_logits must be one dimensional")
    full = np.concatenate([free_logits, np.zeros(1, dtype=np.float64)])
    return softmax(full)


def central_difference_step() -> float:
    """Step size for a central difference of a smooth function.

    The truncation error of a central difference falls as O(h^2) while the
    round-off error grows as O(eps/h), so the total is minimised near
    h = eps**(1/3). This is derived, not tuned: no step size is chosen to make a
    particular error figure appear.
    """
    return float(np.finfo(np.float64).eps ** (1.0 / 3.0))


def finite_difference_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    step: float | None = None,
) -> np.ndarray:
    """Central-difference Jacobian of ``func`` at ``x``.

    Column j holds d func / d x_j. Applied to an analytic gradient this yields a
    numerical Hessian.
    """
    x = np.asarray(x, dtype=np.float64)
    if step is None:
        step = central_difference_step()

    probe = np.asarray(func(x), dtype=np.float64)
    jacobian = np.zeros((probe.size, x.size), dtype=np.float64)
    for j in range(x.size):
        offset = np.zeros_like(x)
        offset[j] = step
        forward = np.asarray(func(x + offset), dtype=np.float64)
        backward = np.asarray(func(x - offset), dtype=np.float64)
        jacobian[:, j] = (forward - backward) / (2.0 * step)
    return jacobian


class NewtonResult:
    """Outcome of a Newton run.

    Attributes
    ----------
    x
        The final parameter vector.
    iterations
        Number of Newton *updates* actually applied. A method that meets the
        stopping criterion at the starting point reports 0.
    gradient_norm
        Infinity norm of the exact gradient at ``x``.
    converged
        Whether the stopping criterion was met within ``max_iterations``.
    history
        One ``(iteration, objective, gradient_norm)`` tuple per evaluation,
        including the starting point.
    """

    def __init__(self, x, iterations, gradient_norm, converged, history):
        self.x = x
        self.iterations = iterations
        self.gradient_norm = gradient_norm
        self.converged = converged
        self.history = history

    def __repr__(self) -> str:
        return (
            f"NewtonResult(iterations={self.iterations}, "
            f"gradient_norm={self.gradient_norm:.3e}, converged={self.converged})"
        )


def newton_descent(
    x0: np.ndarray,
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    system_fn: Callable[[np.ndarray], np.ndarray],
    objective_fn: Callable[[np.ndarray], float],
    tolerance: float,
    max_iterations: int,
) -> NewtonResult:
    """Undamped Newton iteration ``system(x) @ step = -gradient(x)``.

    ``gradient_fn`` must always return the EXACT gradient. ``system_fn`` returns
    the matrix a given method chooses to solve against: the exact Hessian, its
    diagonal, or a diagonal bound. Convergence is therefore always measured on
    the exact gradient, so every method is held to an identical standard and the
    only thing that varies is the quality of the curvature model.

    Stopping criterion: ``max(abs(gradient)) <= tolerance`` (infinity norm),
    checked before each update.

    The system is solved, never inverted. A singular system raises rather than
    being silently regularised, because hiding singularity is exactly the failure
    mode this research is about.
    """
    x = np.array(x0, dtype=np.float64)
    history = []

    for iteration in range(max_iterations + 1):
        gradient = np.asarray(gradient_fn(x), dtype=np.float64)
        gradient_norm = float(np.max(np.abs(gradient)))
        history.append((iteration, float(objective_fn(x)), gradient_norm))

        if gradient_norm <= tolerance:
            return NewtonResult(x, iteration, gradient_norm, True, history)
        if iteration == max_iterations:
            return NewtonResult(x, iteration, gradient_norm, False, history)

        system = np.asarray(system_fn(x), dtype=np.float64)
        step = np.linalg.solve(system, -gradient)
        x = x + step

    raise AssertionError("unreachable")
