"""
Faithful reproduction of the experiment in XGBoost issue #12278.

Purpose
-------
This script answers one question only: "can the behaviour reported upstream be
reproduced?" It is NOT an attempt to redesign or quietly correct the upstream
experiment. The canonical, independently derived mathematics lives in
``exact_multiclass_hessian.py``; the two files are kept separate on purpose.

Setup
-----
Covertype, first 5,000 samples, 7 classes, intercept-only multinomial model.
Because the model has no features, every row shares one probability vector and
the optimum is available in closed form, which gives a check that does not
depend on the optimiser at all:

    z_i = log(q_i / q_{K-1})      with q the empirical class proportions

The Newton system uses the K-1 reference-class parameterization (last class
pinned to logit 0); see ``exact_multiclass_hessian.py`` for why the
K-dimensional Hessian is singular and cannot be used directly.

KNOWN OBJECTIVE-EXPRESSION MISMATCH (preserved deliberately)
------------------------------------------------------------
The upstream experiment's objective is written in a binary, one-vs-rest style,
summed over every entry of the one-hot label matrix:

    L_upstream = -sum_{n,k} [ Y_nk * log(p_nk) + (1 - Y_nk) * log(1 - p_nk) ]

while the gradient and Hessian it uses are the standard multinomial ones:

    g = p - y                 H = diag(p) - p p^T

These are inconsistent: the multinomial gradient is not the derivative of that
binary-style expression. The true multinomial objective is

    L_multinomial = -sum_{n,k} Y_nk * log(p_nk)

This script reports BOTH, clearly labelled, and does not silently substitute one
for the other. The distinction matters for reproducibility: the upstream
objective value can only be reproduced using the upstream expression.

The reported optimum is unaffected. The optimiser is driven by the multinomial
gradient, whose root is the closed form above, so the objective expression acts
as a diagnostic readout here rather than as the thing being minimised. That is
itself part of what is being documented.

Run:
    python research/reproduce_issue_12278.py

Requires: numpy, scikit-learn (for the Covertype fetcher). The dataset is cached
by scikit-learn after the first download.
"""

from __future__ import annotations

import numpy as np

from multinomial import (
    finite_difference_jacobian,
    newton_descent,
    reference_probabilities,
)

# --------------------------------------------------------------------------- #
# Experiment constants
# --------------------------------------------------------------------------- #

N_SAMPLES = 5000
N_CLASSES = 7

# Stopping criterion applied identically to both methods: max|exact gradient|.
# Note this gradient is the SUM over rows, so it carries a factor of n. A given
# numeric tolerance is therefore n times stricter here than the same number
# applied to a per-sample mean gradient.
TOLERANCE = 1e-10
MAX_ITERATIONS = 20000

# Iteration counts are meaningless without the tolerance that produced them, so
# both methods are also run across a sweep. See the note printed with the table.
TOLERANCE_SWEEP = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)

SEPARATOR = "-" * 60


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_class_counts() -> tuple[np.ndarray, int]:
    """Class counts of the first N_SAMPLES Covertype rows.

    The slice is taken in the dataset's natural order with no shuffling, so the
    result is deterministic and no seed is required.
    """
    try:
        from sklearn.datasets import fetch_covtype
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "scikit-learn is required for the Covertype fetcher: pip install scikit-learn"
        ) from error

    dataset = fetch_covtype()
    targets = np.asarray(dataset.target, dtype=np.int64)[:N_SAMPLES]

    # Covertype labels are 1..7; shift to 0..6.
    labels = targets - 1
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        raise AssertionError(f"unexpected label range: [{labels.min()}, {labels.max()}]")

    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float64)
    if int(counts.sum()) != N_SAMPLES:
        raise AssertionError("class counts do not sum to the sample count")
    if np.any(counts == 0):
        raise AssertionError(
            "a class is absent from the slice; the reference-class parameterization "
            "would be degenerate"
        )
    return counts, int(labels.size)


# --------------------------------------------------------------------------- #
# Objectives. Both are reported; see the module docstring.
# --------------------------------------------------------------------------- #


def log_probabilities(free_logits: np.ndarray) -> np.ndarray:
    """Stable log p for the K-1 free-logit parameterization."""
    full = np.concatenate([np.asarray(free_logits, dtype=np.float64), np.zeros(1)])
    shift = np.max(full)
    log_sum_exp = shift + np.log(np.sum(np.exp(full - shift)))
    return full - log_sum_exp


def log_complement_probabilities(log_p: np.ndarray) -> np.ndarray:
    """Stable log(1 - p_k), computed as log(sum_{j != k} p_j).

    Evaluating log(1 - p) directly loses precision as p approaches 1 and hits
    log(0) at p == 1. Summing the complementary probabilities instead is exact
    for every valid distribution and needs no epsilon.
    """
    log_p = np.asarray(log_p, dtype=np.float64)
    out = np.empty_like(log_p)
    for k in range(log_p.size):
        others = np.delete(log_p, k)
        shift = np.max(others)
        out[k] = shift + np.log(np.sum(np.exp(others - shift)))
    return out


def upstream_objective(free_logits: np.ndarray, counts: np.ndarray) -> float:
    """The upstream binary-style expression, reproduced as written.

        -sum_{n,k} [ Y_nk log p_k + (1 - Y_nk) log(1 - p_k) ]

    For an intercept-only model every row shares p, so the sum over rows reduces
    to counts for the Y term and (n - counts) for the (1 - Y) term.
    """
    log_p = log_probabilities(free_logits)
    log_1_minus_p = log_complement_probabilities(log_p)
    n = float(np.sum(counts))
    return float(-(np.sum(counts * log_p) + np.sum((n - counts) * log_1_minus_p)))


def multinomial_objective(free_logits: np.ndarray, counts: np.ndarray) -> float:
    """The standard multinomial negative log-likelihood, -sum_k n_k log p_k.

    This is the objective whose gradient actually is ``p - y``.
    """
    log_p = log_probabilities(free_logits)
    return float(-np.sum(counts * log_p))


# --------------------------------------------------------------------------- #
# Multinomial gradient and curvature over the K-1 free coordinates
# --------------------------------------------------------------------------- #


def gradient(free_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Exact multinomial gradient: n * p_i - n_i, for i < K-1.

    Sign convention: this is the gradient of the NEGATIVE log-likelihood, i.e.
    ``p - y`` summed over rows. The upstream notebook writes ``Y - p``, which is
    the same quantity with the opposite sign because it ascends the likelihood
    rather than descending the loss. The Newton step is identical either way as
    long as the sign is used consistently.
    """
    p = reference_probabilities(free_logits)
    n = float(np.sum(counts))
    return n * p[:-1] - counts[:-1]


def exact_hessian(free_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Exact dense Hessian: n * (diag(p) - p p^T) over the free coordinates."""
    p = reference_probabilities(free_logits)
    n = float(np.sum(counts))
    free_p = p[:-1]
    return n * (np.diag(free_p) - np.outer(free_p, free_p))


def diagonal_bound(free_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """XGBoost-style diagonal upper bound: n * 2 * p_i * (1 - p_i).

    This is the approximation issue #12278 is about. It is a valid majoriser of
    the exact Hessian, so steps are safe without a line search, but it discards
    the off-diagonal coupling and halves the effective step length.
    """
    p = reference_probabilities(free_logits)
    n = float(np.sum(counts))
    free_p = p[:-1]
    return np.diag(2.0 * n * free_p * (1.0 - free_p))


def closed_form_solution(counts: np.ndarray) -> np.ndarray:
    """z_i = log(q_i / q_{K-1}); the optimum of an intercept-only model."""
    proportions = counts / float(np.sum(counts))
    return np.log(proportions[:-1] / proportions[-1])


# --------------------------------------------------------------------------- #
# Cross-checks
# --------------------------------------------------------------------------- #


def cross_check_curvature(free_logits: np.ndarray, counts: np.ndarray) -> dict:
    """Analytic Hessian vs finite differences, symmetry, definiteness."""
    analytic = exact_hessian(free_logits, counts)
    numerical = finite_difference_jacobian(lambda z: gradient(z, counts), free_logits)

    fd_error = float(np.max(np.abs(analytic - numerical)))
    # Relative, because the dataset Hessian is scaled by n = 5000.
    scale = float(np.max(np.abs(analytic)))
    symmetry_error = float(np.max(np.abs(analytic - analytic.T)))
    eigenvalues = np.linalg.eigvalsh(analytic)

    if symmetry_error > 1e-12 * scale:
        raise AssertionError(f"Hessian is not symmetric: {symmetry_error:.3e}")
    if eigenvalues[0] <= 0.0:
        raise AssertionError("free Hessian should be positive definite")
    if fd_error > 1e-4 * scale:
        raise AssertionError(f"Hessian disagrees with finite differences: {fd_error:.3e}")

    return {
        "fd_error": fd_error,
        "fd_error_relative": fd_error / scale,
        "symmetry_error": symmetry_error,
        "smallest_eigenvalue": float(eigenvalues[0]),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
    }


def run_method(name, system_fn, counts, tolerance=TOLERANCE):
    start = np.zeros(counts.size - 1, dtype=np.float64)
    result = newton_descent(
        x0=start,
        gradient_fn=lambda z: gradient(z, counts),
        system_fn=lambda z: system_fn(z, counts),
        objective_fn=lambda z: multinomial_objective(z, counts),
        tolerance=tolerance,
        max_iterations=MAX_ITERATIONS,
    )
    if not result.converged:
        raise AssertionError(f"{name} did not converge within {MAX_ITERATIONS} iterations")
    return result


def run_tolerance_sweep(counts) -> list:
    """Iteration counts for both methods across a range of stopping tolerances.

    A single iteration count is not a reportable result on its own, because it
    depends entirely on how hard the stopping criterion is. Sweeping it separates
    the two convergence regimes: Newton on the exact Hessian converges
    quadratically, so demanding more decades of accuracy costs almost nothing,
    while the diagonal bound converges linearly, so its cost grows roughly in
    proportion to the number of decades demanded.
    """
    rows = []
    for tolerance in TOLERANCE_SWEEP:
        exact = run_method("exact", exact_hessian, counts, tolerance)
        bounded = run_method("diagonal_bound", diagonal_bound, counts, tolerance)
        rows.append((tolerance, exact.iterations, bounded.iterations))
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    counts, n = load_class_counts()
    proportions = counts / float(n)

    exact = run_method("exact", exact_hessian, counts)
    bounded = run_method("diagonal_bound", diagonal_bound, counts)

    closed_form = closed_form_solution(counts)
    solution_error = float(np.max(np.abs(exact.x - closed_form)))

    final_probabilities = reference_probabilities(exact.x)
    probability_error = float(np.max(np.abs(final_probabilities - proportions)))

    curvature = cross_check_curvature(exact.x, counts)

    upstream_at_optimum = upstream_objective(exact.x, counts)
    multinomial_at_optimum = multinomial_objective(exact.x, counts)

    print(SEPARATOR)
    print("XGBoost #12278 Reproduction")
    print(SEPARATOR)
    print("Dataset                    : Covertype (scikit-learn fetch_covtype)")
    print(f"Samples (n)                : {n}")
    print(f"Classes (K)                : {N_CLASSES}")
    print(f"Free coordinates (K-1)     : {N_CLASSES - 1}")
    print("Slice                      : first n rows, natural order, no shuffle")
    print()
    print(f"Class counts               : {counts.astype(int).tolist()}")
    print(f"Class proportions          : {proportions}")
    print()

    print("Closed-form optimum (intercept-only model)")
    print(f"  z_i = log(q_i / q_K-1)   : {closed_form}")
    print()

    print("Exact Newton (dense Hessian)")
    print(f"  iterations               : {exact.iterations}")
    print(f"  final gradient norm      : {exact.gradient_norm:.6e}")
    print(f"  free logits              : {exact.x}")
    print(f"  max diff vs closed form  : {solution_error:.3e}")
    print(f"  objective (multinomial)  : {multinomial_at_optimum:.9f}")
    print(f"  objective (upstream expr): {upstream_at_optimum:.9f}")
    print()

    print("Diagonal upper bound (2 * p * (1 - p))")
    print(f"  iterations               : {bounded.iterations}")
    print(f"  final gradient norm      : {bounded.gradient_norm:.6e}")
    print(f"  objective (multinomial)  : {multinomial_objective(bounded.x, counts):.9f}")
    print(f"  max diff vs closed form  : {float(np.max(np.abs(bounded.x - closed_form))):.3e}")
    print()

    ratio = bounded.iterations / exact.iterations if exact.iterations else float("inf")
    print(f"Iteration ratio            : {ratio:.1f}x  (bound / exact)")
    print(f"Stopping criterion         : max|exact gradient| <= {TOLERANCE:.1e}")
    print()

    print("Sensitivity of the iteration counts to the stopping tolerance")
    print(f"  {'tolerance':>10} {'exact':>8} {'bound':>8} {'ratio':>8}")
    for tolerance, exact_iterations, bound_iterations in run_tolerance_sweep(counts):
        sweep_ratio = bound_iterations / exact_iterations if exact_iterations else float("inf")
        print(f"  {tolerance:>10.0e} {exact_iterations:>8} {bound_iterations:>8} "
              f"{sweep_ratio:>8.0f}")
    print("  Newton on the exact Hessian is near tolerance-independent (quadratic")
    print("  convergence); the diagonal bound grows roughly linearly in the number of")
    print("  decades demanded. Any single iteration count is only meaningful together")
    print("  with the tolerance that produced it.")
    print()

    print("Final probabilities")
    print(f"  fitted                   : {final_probabilities}")
    print(f"  empirical proportions    : {proportions}")
    print(f"  max abs difference       : {probability_error:.3e}")
    print()

    print("Curvature diagnostics at the optimum")
    print(f"  finite-difference error  : {curvature['fd_error']:.3e} "
          f"(relative {curvature['fd_error_relative']:.3e})")
    print(f"  symmetry error           : {curvature['symmetry_error']:.3e}")
    print(f"  smallest eigenvalue      : {curvature['smallest_eigenvalue']:.6e}  (> 0)")
    print(f"  largest eigenvalue       : {curvature['largest_eigenvalue']:.6e}")
    print(f"  condition number         : {curvature['condition_number']:.6f}")
    print()

    print("Objective-expression mismatch (documented, not corrected)")
    print("  The upstream expression sums a binary one-vs-rest log-likelihood")
    print("    -sum[ Y log p + (1-Y) log(1-p) ]")
    print("  while the gradient/Hessian used are multinomial (p - y, diag(p) - p p^T).")
    print("  The multinomial gradient is not the derivative of that expression, so the")
    print("  two objective columns above are reported separately and neither is")
    print("  substituted for the other.")
    print(f"  upstream expression value: {upstream_at_optimum:.9f}")
    print(f"  multinomial NLL value    : {multinomial_at_optimum:.9f}")
    print(SEPARATOR)

    if probability_error > 1e-9:
        raise AssertionError("fitted probabilities do not match the empirical proportions")
    if solution_error > 1e-9:
        raise AssertionError("Newton solution does not match the closed form")
    print("All cross-checks passed.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
