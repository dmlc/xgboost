"""
Research vs production crosscheck for `multi_hessian=exact`.

The research scripts solve an intercept-only multinomial model with a pure Newton method.
XGBoost solves the same model by additive boosting with a learning rate. The two are NOT the
same algorithm, so this script does not compare iteration counts or trajectories -- it
compares the **fixed point**: both should land on the empirical class proportions, which is
the optimum of an intercept-only multinomial model.

What is compared
----------------
    empirical class proportions      (data)
    research Newton solution         (exact_multiclass_hessian.py mathematics)
    XGBoost exact-mode prediction    (production training path)
    multinomial negative log-likelihood at each solution
    gradient infinity norm at each solution

Run with the package built from this source tree:

    PYTHONPATH=python-package python research/validate_xgboost_exact.py

Requires numpy and xgboost; the Covertype section additionally needs scikit-learn and skips
cleanly without it.
"""

from __future__ import annotations

import numpy as np

from multinomial import reference_probabilities, softmax

try:
    import xgboost as xgb
except ImportError as error:  # pragma: no cover - environment dependent
    raise SystemExit(
        "xgboost is not importable. Build the library and run with "
        "PYTHONPATH=python-package."
    ) from error

SEPARATOR = "-" * 72

# Intercept-only training needs no depth; one constant feature makes every tree a stump.
BASE_PARAMS = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "device": "cpu",
    "multi_strategy": "multi_output_tree",
    "multi_hessian": "exact",
    "max_depth": 1,
    "eta": 1.0,
    "lambda": 0.0,
    "base_score": 0.5,
}
ROUNDS = 80


def multinomial_nll(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """-sum_k n_k log p_k, the objective an intercept-only multinomial model minimises."""
    clipped = np.clip(probabilities, 1e-300, None)
    return float(-np.sum(counts * np.log(clipped)))


def gradient_norm(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """Infinity norm of the exact gradient over the K-1 free coordinates."""
    n = float(np.sum(counts))
    free = n * probabilities[:-1] - counts[:-1]
    return float(np.max(np.abs(free)))


def research_solution(counts: np.ndarray) -> np.ndarray:
    """Closed-form optimum of the intercept-only model: z_i = log(q_i / q_{K-1})."""
    proportions = counts / float(np.sum(counts))
    free_logits = np.log(proportions[:-1] / proportions[-1])
    return reference_probabilities(free_logits)


def xgboost_solution(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Train exact mode on a single constant feature and return the fitted distribution."""
    x = np.ones((labels.size, 1), dtype=np.float32)
    dtrain = xgb.DMatrix(x, label=labels.astype(np.float32))
    params = dict(BASE_PARAMS)
    params["num_class"] = int(n_classes)
    booster = xgb.train(params, dtrain, num_boost_round=ROUNDS)
    predictions = booster.predict(dtrain)
    # Intercept-only, so every row shares one distribution; verify rather than assume.
    spread = float(np.max(np.abs(predictions - predictions[0])))
    if spread > 1e-5:
        raise AssertionError(f"rows disagree in an intercept-only model: {spread:.3e}")
    return predictions[0].astype(np.float64)


def report(name: str, counts: np.ndarray) -> None:
    counts = counts.astype(np.float64)
    n_classes = counts.size
    proportions = counts / float(np.sum(counts))

    research = research_solution(counts)
    production = xgboost_solution(
        np.concatenate([np.full(int(c), k) for k, c in enumerate(counts)]), n_classes
    )

    print(SEPARATOR)
    print(f"{name}: n = {int(np.sum(counts))}, K = {n_classes}")
    print(SEPARATOR)
    print(f"  class counts              : {counts.astype(int).tolist()}")
    print(f"  empirical proportions     : {np.round(proportions, 6)}")
    print(f"  research Newton solution  : {np.round(research, 6)}")
    print(f"  XGBoost exact prediction  : {np.round(production, 6)}")
    print()
    print(f"  max |research - empirical|: {np.max(np.abs(research - proportions)):.3e}")
    print(f"  max |xgboost  - empirical|: {np.max(np.abs(production - proportions)):.3e}")
    print(f"  max |xgboost  - research |: {np.max(np.abs(production - research)):.3e}")
    print()
    print(f"  multinomial NLL, research : {multinomial_nll(counts, research):.9f}")
    print(f"  multinomial NLL, xgboost  : {multinomial_nll(counts, production):.9f}")
    print(f"  gradient norm,   research : {gradient_norm(counts, research):.3e}")
    print(f"  gradient norm,   xgboost  : {gradient_norm(counts, production):.3e}")
    print()


def synthetic_cases() -> None:
    """Unbalanced intercept problems at several K."""
    for n_classes in (2, 3, 7):
        parts = np.arange(1, n_classes + 1, dtype=np.float64)
        counts = np.round(900 * parts / parts.sum()).astype(np.float64)
        counts[-1] += 900 - counts.sum()
        report(f"synthetic K={n_classes}", counts)


def covertype_case() -> None:
    """The issue #12278 setup: first 5,000 Covertype rows, 7 classes, intercept only."""
    try:
        from sklearn.datasets import fetch_covtype
    except ImportError:
        print(SEPARATOR)
        print("Covertype: skipped (scikit-learn is not installed)")
        print(SEPARATOR)
        return

    targets = np.asarray(fetch_covtype().target, dtype=np.int64)[:5000] - 1
    counts = np.bincount(targets, minlength=7).astype(np.float64)
    if np.any(counts == 0):
        print("Covertype: skipped (a class is absent from the slice)")
        return
    report("Covertype first 5000", counts)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    print()
    print("Research (pure Newton) vs production (XGBoost boosting) -- comparing fixed points,")
    print("not trajectories. XGBoost boosts additively with a learning rate, so its iteration")
    print("count is not comparable to the research solver's and is deliberately not reported.")
    print()
    synthetic_cases()
    covertype_case()
    print(SEPARATOR)
    print("Both solvers target the same optimum: the empirical class proportions.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
