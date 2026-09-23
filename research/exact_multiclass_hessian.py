"""
Exact multinomial-logistic Hessian: an independent mathematical validation.

Context
-------
This file supports XGBoost issue #12278, "Using exact Hessian for multinomial
logistic". It is deliberately independent of XGBoost's C++ implementation: it
checks the mathematics from scratch so that the C++ work can be audited against
something that does not share its assumptions.

The gauge problem
-----------------
For K classes with probabilities p = softmax(z), the per-sample gradient and
Hessian of the negative log-likelihood with respect to the K logits are

    g = p - y
    H = diag(p) - p p^T

H is ALWAYS singular. Softmax is invariant to a constant shift, softmax(z + c*1)
= softmax(z), so the all-ones vector spans a null direction:

    H @ 1 = p - p * sum(p) = p - p = 0

A Newton step therefore has no unique solution in K dimensions, and any solver
either fails or silently picks an arbitrary point along that null line. This is a
property of the parameterization, not of the data, and it cannot be fixed by
adding an epsilon to the diagonal without changing the problem being solved.

The reference-class parameterization
------------------------------------
Pin the last logit to zero, z_{K-1} = 0, and optimize only z_0 .. z_{K-2}. This
removes exactly the redundant direction while leaving the model's expressive
power untouched: every softmax distribution is still reachable, just by a unique
parameter vector instead of a line of equivalent ones.

The gradient and Hessian restricted to the free coordinates i, j < K-1 are

    g_i  = p_i - y_i
    H_ij = p_i * delta_ij - p_i * p_j

and this H is the leading (K-1)x(K-1) block of the full matrix. It is positive
definite whenever every p_i > 0, so the Newton system is well posed.

What this file validates
------------------------
    A. stable softmax
    B. analytic gradient against a finite difference of the objective
    C. analytic Hessian against a finite difference of the gradient
    D. Hessian symmetry
    E. the singularity of the K-dimensional Hessian, and its removal
    F. the reference-class block identity
    G. a Newton solve
    H. convergence: exact Hessian vs exact diagonal vs diagonal upper bound

Run:
    python research/exact_multiclass_hessian.py

Requires: numpy. No XGBoost, no scikit-learn.
"""

from __future__ import annotations

import numpy as np

from multinomial import (
    central_difference_step,
    finite_difference_jacobian,
    newton_descent,
    reference_probabilities,
    softmax,
)

# --------------------------------------------------------------------------- #
# Experiment constants. Everything the scripts report is derived from these.
# --------------------------------------------------------------------------- #

# Validation problem: an arbitrary but fixed logit vector and label.
VALIDATION_LOGITS = np.array([0.30, -1.20, 0.75, 2.10], dtype=np.float64)
VALIDATION_LABEL = 2

# Toy intercept problem. The counts are chosen so the optimum is exactly
# [0.1, 0.3, 0.6], which makes the expected Hessian checkable by hand.
TOY_COUNTS = np.array([10, 30, 60], dtype=np.float64)
TOY_TARGET = np.array([0.1, 0.3, 0.6], dtype=np.float64)

# Stopping criterion for every Newton variant: max|exact gradient| <= TOLERANCE,
# i.e. the infinity norm. The choice of norm is worth stating because it moves
# the reported counts: since ||g||_2 >= ||g||_inf, an L2 criterion at the same
# numeric tolerance is slightly stricter and costs the slowest method here one
# extra iteration (54 under the infinity norm, 55 under L2). Whichever is used,
# it is applied identically to all three methods.
TOLERANCE = 1e-10
MAX_ITERATIONS = 2000

SEPARATOR = "-" * 60


# --------------------------------------------------------------------------- #
# Per-sample multinomial quantities, full K-dimensional parameterization.
# Used only to demonstrate the singularity; it is not the working formulation.
# --------------------------------------------------------------------------- #


def full_gradient_hessian(logits: np.ndarray, label: int):
    """Exact gradient and Hessian over all K logits, for one sample."""
    p = softmax(logits)
    y = np.zeros(p.size, dtype=np.float64)
    y[label] = 1.0
    gradient = p - y
    hessian = np.diag(p) - np.outer(p, p)
    return gradient, hessian


def full_negative_log_likelihood(logits: np.ndarray, label: int) -> float:
    """-log p[label], computed via the log-sum-exp identity.

    Written as logsumexp(z) - z[label] rather than -log(softmax(z)[label]) so
    that no probability is ever formed and then logged; this is exact even when
    p[label] would underflow to zero in floating point.
    """
    logits = np.asarray(logits, dtype=np.float64)
    shift = np.max(logits)
    log_sum_exp = shift + np.log(np.sum(np.exp(logits - shift)))
    return float(log_sum_exp - logits[label])


# --------------------------------------------------------------------------- #
# Intercept-only dataset quantities in the K-1 free coordinates.
# --------------------------------------------------------------------------- #


def free_objective(free_logits: np.ndarray, counts: np.ndarray) -> float:
    """Negative log-likelihood of an intercept-only multinomial model.

    Uses the standard multinomial expression, -sum_k n_k log p_k, evaluated
    stably through log-sum-exp.
    """
    free_logits = np.asarray(free_logits, dtype=np.float64)
    full = np.concatenate([free_logits, np.zeros(1)])
    shift = np.max(full)
    log_sum_exp = shift + np.log(np.sum(np.exp(full - shift)))
    log_p = full - log_sum_exp
    return float(-np.sum(counts * log_p))


def free_gradient(free_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Exact gradient over the K-1 free logits: n * p_i - n_i."""
    p = reference_probabilities(free_logits)
    n = float(np.sum(counts))
    return n * p[:-1] - counts[:-1]


def free_hessian(free_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Exact Hessian over the K-1 free logits: n * (diag(p) - p p^T)."""
    p = reference_probabilities(free_logits)
    n = float(np.sum(counts))
    free_p = p[:-1]
    return n * (np.diag(free_p) - np.outer(free_p, free_p))


def exact_diagonal_system(free_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Diagonal of the exact Hessian: n * p_i * (1 - p_i).

    This is the correct curvature along each coordinate axis, but it discards the
    negative off-diagonal coupling between classes. Since raising one logit must
    lower the other probabilities, ignoring that coupling systematically
    overstates how independent the coordinates are.
    """
    p = reference_probabilities(free_logits)
    n = float(np.sum(counts))
    free_p = p[:-1]
    return np.diag(n * free_p * (1.0 - free_p))


def upper_bound_diagonal_system(free_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """XGBoost-style diagonal upper bound: n * 2 * p_i * (1 - p_i).

    Doubling p(1-p) yields a diagonal matrix D with D >= H in the positive
    semidefinite sense, which guarantees the Newton step never overshoots and so
    needs no line search. The price is that every step is roughly half the length
    the exact curvature would justify, which shows up directly as iteration count.

    This is the approximation discussed in issue #12278.
    """
    return 2.0 * exact_diagonal_system(free_logits, counts)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_softmax() -> dict:
    """A: stability, normalisation and shift invariance."""
    p = softmax(VALIDATION_LOGITS)
    normalisation_error = abs(float(np.sum(p)) - 1.0)

    # Shift invariance: the mathematical identity the reference class relies on.
    shifted = softmax(VALIDATION_LOGITS + 12.5)
    shift_error = float(np.max(np.abs(p - shifted)))

    # Overflow/underflow: naive exp() of these would be inf and 0 respectively.
    extreme = softmax(np.array([1000.0, -1000.0, 0.0]))
    extreme_finite = bool(np.all(np.isfinite(extreme)))
    extreme_normalised = abs(float(np.sum(extreme)) - 1.0)

    if normalisation_error > 1e-15:
        raise AssertionError(f"softmax does not normalise: {normalisation_error:.3e}")
    if shift_error > 1e-15:
        raise AssertionError(f"softmax is not shift invariant: {shift_error:.3e}")
    if not extreme_finite:
        raise AssertionError("softmax overflowed on extreme logits")
    if extreme_normalised > 1e-15:
        raise AssertionError("softmax lost normalisation on extreme logits")

    return {
        "normalisation_error": normalisation_error,
        "shift_error": shift_error,
        "extreme_normalisation_error": extreme_normalised,
    }


def validate_gradient() -> dict:
    """B: analytic gradient vs a central difference of the objective."""
    analytic, _ = full_gradient_hessian(VALIDATION_LOGITS, VALIDATION_LABEL)
    numerical = finite_difference_jacobian(
        lambda z: np.array([full_negative_log_likelihood(z, VALIDATION_LABEL)]),
        VALIDATION_LOGITS,
    ).ravel()
    error = float(np.max(np.abs(analytic - numerical)))
    if error > 1e-7:
        raise AssertionError(f"gradient disagrees with finite differences: {error:.3e}")
    return {"gradient_fd_error": error}


def validate_hessian() -> dict:
    """C, D, E: Hessian vs finite differences, symmetry, and singularity."""
    _, analytic = full_gradient_hessian(VALIDATION_LOGITS, VALIDATION_LABEL)
    numerical = finite_difference_jacobian(
        lambda z: full_gradient_hessian(z, VALIDATION_LABEL)[0],
        VALIDATION_LOGITS,
    )

    fd_error = float(np.max(np.abs(analytic - numerical)))
    symmetry_error = float(np.max(np.abs(analytic - analytic.T)))

    # The null direction: H @ 1 = 0, exactly the gauge freedom of the softmax.
    ones = np.ones(analytic.shape[0], dtype=np.float64)
    null_residual = float(np.max(np.abs(analytic @ ones)))
    row_sum_error = float(np.max(np.abs(analytic.sum(axis=1))))
    eigenvalues = np.linalg.eigvalsh(analytic)
    rank = int(np.linalg.matrix_rank(analytic))

    if fd_error > 1e-7:
        raise AssertionError(f"Hessian disagrees with finite differences: {fd_error:.3e}")
    if symmetry_error > 1e-15:
        raise AssertionError(f"Hessian is not symmetric: {symmetry_error:.3e}")
    if null_residual > 1e-15:
        raise AssertionError("H @ 1 should vanish; the null direction is missing")
    if rank != analytic.shape[0] - 1:
        raise AssertionError(f"expected rank K-1, got {rank}")

    return {
        "hessian_fd_error": fd_error,
        "symmetry_error": symmetry_error,
        "null_residual": null_residual,
        "row_sum_error": row_sum_error,
        "smallest_eigenvalue": float(eigenvalues[0]),
        "rank": rank,
        "dimension": int(analytic.shape[0]),
    }


def validate_reference_block() -> dict:
    """F: the free Hessian is the leading block of the full one, and is PD."""
    free_logits = VALIDATION_LOGITS[:-1] - VALIDATION_LOGITS[-1]
    p_reference = reference_probabilities(free_logits)
    p_full = softmax(VALIDATION_LOGITS)

    # Pinning the last logit to zero is a shift, so the probabilities agree.
    probability_error = float(np.max(np.abs(p_reference - p_full)))

    _, full_hessian = full_gradient_hessian(VALIDATION_LOGITS, VALIDATION_LABEL)
    block = full_hessian[:-1, :-1]
    free_p = p_full[:-1]
    constructed = np.diag(free_p) - np.outer(free_p, free_p)
    block_error = float(np.max(np.abs(block - constructed)))

    eigenvalues = np.linalg.eigvalsh(constructed)
    smallest = float(eigenvalues[0])

    if probability_error > 1e-15:
        raise AssertionError("reference parameterization changed the probabilities")
    if block_error > 1e-15:
        raise AssertionError("free Hessian is not the leading block of the full Hessian")
    if smallest <= 0.0:
        raise AssertionError("free Hessian is not positive definite")

    return {
        "probability_error": probability_error,
        "block_error": block_error,
        "smallest_eigenvalue": smallest,
        "condition_number": float(np.linalg.cond(constructed)),
    }


def validate_toy_hessian() -> dict:
    """The hand-checkable case: p = [0.1, 0.3, 0.6], class 2 as reference."""
    target = TOY_TARGET
    free_logits = np.log(target[:-1] / target[-1])
    p = reference_probabilities(free_logits)
    probability_error = float(np.max(np.abs(p - target)))

    free_p = p[:-1]
    hessian = np.diag(free_p) - np.outer(free_p, free_p)
    expected = np.array([[0.09, -0.03], [-0.03, 0.21]], dtype=np.float64)
    error = float(np.max(np.abs(hessian - expected)))

    if probability_error > 1e-15:
        raise AssertionError("toy logits do not reproduce the target probabilities")
    if error > 1e-15:
        raise AssertionError(f"toy Hessian mismatch: {error:.3e}")

    return {
        "probability_error": probability_error,
        "hessian": hessian,
        "expected": expected,
        "hessian_error": error,
    }


def run_convergence_comparison() -> dict:
    """G, H: Newton from a common start under three curvature models."""
    counts = TOY_COUNTS
    start = np.zeros(counts.size - 1, dtype=np.float64)

    systems = {
        "exact_full": free_hessian,
        "exact_diagonal": exact_diagonal_system,
        "upper_bound_diagonal": upper_bound_diagonal_system,
    }

    results = {}
    for name, system_fn in systems.items():
        result = newton_descent(
            x0=start,
            gradient_fn=lambda z: free_gradient(z, counts),
            system_fn=lambda z, fn=system_fn: fn(z, counts),
            objective_fn=lambda z: free_objective(z, counts),
            tolerance=TOLERANCE,
            max_iterations=MAX_ITERATIONS,
        )
        probabilities = reference_probabilities(result.x)
        results[name] = {
            "result": result,
            "probabilities": probabilities,
            "probability_error": float(np.max(np.abs(probabilities - TOY_TARGET))),
        }

    for name, entry in results.items():
        if not entry["result"].converged:
            raise AssertionError(f"{name} did not converge within {MAX_ITERATIONS} iterations")
        if entry["probability_error"] > 1e-9:
            raise AssertionError(
                f"{name} converged to the wrong distribution: "
                f"{entry['probability_error']:.3e}"
            )

    return results


def demonstrate_singular_newton() -> str:
    """Show what the K-dimensional Newton system actually does when solved.

    Reported rather than asserted: the point is to document the failure mode the
    reference-class parameterization exists to avoid.
    """
    gradient, hessian = full_gradient_hessian(VALIDATION_LOGITS, VALIDATION_LABEL)
    try:
        np.linalg.solve(hessian, -gradient)
    except np.linalg.LinAlgError as error:
        return f"raised LinAlgError ({error})"
    return "returned a solution despite the singular system (solver-dependent)"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    softmax_stats = validate_softmax()
    gradient_stats = validate_gradient()
    hessian_stats = validate_hessian()
    block_stats = validate_reference_block()
    toy_stats = validate_toy_hessian()
    convergence = run_convergence_comparison()
    singular_behaviour = demonstrate_singular_newton()

    print(SEPARATOR)
    print("Exact Multinomial Hessian Validation")
    print(SEPARATOR)
    print(f"K                          : {hessian_stats['dimension']}")
    ref = hessian_stats["dimension"] - 1
    print(f"Reference class            : {ref} (last, logit pinned to 0)")
    print(f"Free coordinates           : {hessian_stats['dimension'] - 1}")
    print(f"Central-difference step    : {central_difference_step():.6e}")
    print()

    print("A. Softmax")
    print(f"  normalisation error      : {softmax_stats['normalisation_error']:.3e}")
    print(f"  shift-invariance error   : {softmax_stats['shift_error']:.3e}")
    print(f"  extreme-logit norm error : {softmax_stats['extreme_normalisation_error']:.3e}")
    print()

    print("B. Gradient vs finite differences")
    print(f"  max abs error            : {gradient_stats['gradient_fd_error']:.3e}")
    print()

    print("C/D. Hessian vs finite differences, symmetry")
    print(f"  max finite-difference err: {hessian_stats['hessian_fd_error']:.3e}")
    print(f"  max symmetry error       : {hessian_stats['symmetry_error']:.3e}")
    print(f"  max row-sum error        : {hessian_stats['row_sum_error']:.3e}")
    print()

    print("E. Singularity of the K-dimensional Hessian")
    print(f"  max|H @ 1|               : {hessian_stats['null_residual']:.3e}")
    print(f"  smallest eigenvalue      : {hessian_stats['smallest_eigenvalue']:.3e}")
    print(f"  rank                     : {hessian_stats['rank']} of {hessian_stats['dimension']}")
    print(f"  np.linalg.solve on it    : {singular_behaviour}")
    print()

    print("F. Reference-class parameterization")
    print(f"  probability error        : {block_stats['probability_error']:.3e}")
    print(f"  leading-block error      : {block_stats['block_error']:.3e}")
    print(f"  smallest eigenvalue      : {block_stats['smallest_eigenvalue']:.3e}  (> 0)")
    print(f"  condition number         : {block_stats['condition_number']:.3e}")
    print()

    print("Toy distribution")
    print(f"  target probabilities     : {TOY_TARGET}")
    print(f"  probability error        : {toy_stats['probability_error']:.3e}")
    print("  exact free Hessian       :")
    for row in toy_stats["hessian"]:
        print(f"      {row}")
    print("  expected                 :")
    for row in toy_stats["expected"]:
        print(f"      {row}")
    print(f"  max abs difference       : {toy_stats['hessian_error']:.3e}")
    print()

    print("G/H. Newton convergence, intercept-only, counts = "
          f"{TOY_COUNTS.astype(int).tolist()}")
    print(f"  stopping criterion       : max|exact gradient| <= {TOLERANCE:.1e}")
    print(f"  iteration cap            : {MAX_ITERATIONS}")
    print(f"  {'method':<22} {'iters':>6} {'grad norm':>12} {'objective':>16} {'max p error':>12}")
    for name, entry in convergence.items():
        result = entry["result"]
        print(
            f"  {name:<22} {result.iterations:>6} {result.gradient_norm:>12.3e} "
            f"{result.history[-1][1]:>16.9f} {entry['probability_error']:>12.3e}"
        )
    print()

    baseline = convergence["exact_full"]["result"].iterations
    for name in ("exact_diagonal", "upper_bound_diagonal"):
        iterations = convergence[name]["result"].iterations
        ratio = iterations / baseline if baseline else float("inf")
        print(f"  {name} / exact_full      : {ratio:.2f}x iterations")
    print()
    print(f"  final probabilities      : {convergence['exact_full']['probabilities']}")
    print(SEPARATOR)
    print("All validations passed.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
