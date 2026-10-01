"""
Phase 3D evidence: a small, bounded sanity sweep for the exact-multinomial-Hessian feature.

This is NOT a replacement for ``research/benchmark_v2.py``, which is the authoritative,
validation-only, multi-seed protocol for exact-vs-diagonal accuracy/convergence claims. This
script answers a narrower, structural question instead: does ``multi_hessian=exact`` behave
sanely -- and does the Phase 3B (histogram cache) and Phase 3C (thread-buffer parallelization)
work from this branch actually do what it claims -- across a few scaling dimensions, at a scale
that finishes in well under an hour on a laptop. Every run below uses a tiny dataset (sklearn
``digits``), small round counts, and 2-3 points per dimension: enough to see a trend, not enough
to be a statistically rigorous claim. Do not quote these numbers as benchmark headlines.

Covers, each as its own small experiment:
  - loss vs rounds, loss vs wall time, rounds-to-a-shared-target-loss
  - K scaling (num_class = 2, 5, 10, via class subsets of `digits`)
  - N scaling (training rows = 200, 600, 1200)
  - max_bin scaling (32, 128, 256)
  - thread scaling (1, 2, 4) -- this is the one most relevant to Phase 3C
  - peak memory, illustrated via Phase 3B's own cache budget (tiny vs effectively-unbounded
    max_cached_hist_node on a deeper tree), using process RSS as a coarse proxy

Usage:
    PYTHONPATH=python-package python research/benchmark_phase3d_evidence.py
"""

from __future__ import annotations

import gc
import json
import time

import numpy as np
import psutil
from sklearn.datasets import load_digits

import xgboost as xgb

PROC = psutil.Process()


def rss_mb() -> float:
    gc.collect()
    return PROC.memory_info().rss / (1024.0 * 1024.0)


def peak_wset_mb() -> float:
    """Windows' peak working-set counter: monotonically non-decreasing over the process's
    life, so unlike point-in-time RSS it cannot be hidden by a GC pass that runs between the
    measurement and whenever the peak actually occurred."""
    gc.collect()
    return PROC.memory_info().peak_wset / (1024.0 * 1024.0)


def load(k_classes: int | None = None, n_rows: int | None = None, seed: int = 0):
    d = load_digits()
    x, y = d.data.astype(np.float32), d.target.astype(np.int32)
    if k_classes is not None:
        mask = y < k_classes
        x, y = x[mask], y[mask]
    if n_rows is not None and n_rows < len(y):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=n_rows, replace=False)
        x, y = x[idx], y[idx]
    n_classes = int(y.max()) + 1
    return x, y, n_classes


def params_for(mode: str, n_classes: int, nthread: int = 4, max_bin: int = 256,
              max_cached_hist_node: int | None = None, max_depth: int = 4) -> dict:
    p = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "tree_method": "hist",
        "device": "cpu",
        "multi_strategy": "multi_output_tree",
        "multi_hessian": mode,
        "max_depth": max_depth,
        "eta": 0.3,
        "lambda": 1.0,
        "nthread": nthread,
        "max_bin": max_bin,
        "base_score": 0.5,
    }
    if max_cached_hist_node is not None:
        p["max_cached_hist_node"] = max_cached_hist_node
    return p


class RoundTimer(xgb.callback.TrainingCallback):
    def __init__(self):
        self.elapsed: list[float] = []
        self._t0 = None

    def before_training(self, model):
        self._t0 = time.perf_counter()
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.elapsed.append(time.perf_counter() - self._t0)
        return False


def timed_train(params: dict, dtr, rounds: int, evals=None, evals_result=None):
    timer = RoundTimer()
    t0 = time.perf_counter()
    bst = xgb.train(params, dtr, num_boost_round=rounds, evals=evals or [],
                    evals_result=evals_result if evals_result is not None else {},
                    verbose_eval=False, callbacks=[timer])
    total = time.perf_counter() - t0
    return bst, total, timer.elapsed


# --------------------------------------------------------------------------- 1) loss curves


def section_loss_curves():
    print("\n=== 1) loss vs rounds, loss vs wall time, rounds-to-shared-target ===")
    x, y, k = load()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_tr, n_va = int(0.6 * len(y)), int(0.2 * len(y))
    tr, va = idx[:n_tr], idx[n_tr:n_tr + n_va]
    dtr = xgb.DMatrix(x[tr], label=y[tr])
    dva = xgb.DMatrix(x[va], label=y[va])

    curves = {}
    for mode in ("diagonal", "exact"):
        params = params_for(mode, k)
        result = {}
        bst, total, elapsed = timed_train(params, dtr, rounds=40,
                                          evals=[(dva, "valid")], evals_result=result)
        curves[mode] = {
            "valid_mlogloss": result["valid"]["mlogloss"],
            "elapsed": elapsed,
            "total_seconds": total,
        }
        print(f"  {mode:<9} total={total:.3f}s  final_valid_mlogloss="
              f"{result['valid']['mlogloss'][-1]:.5f}")

    # Rounds-to-target: the worse of the two final losses is a target both modes have reached.
    target = max(curves["diagonal"]["valid_mlogloss"][-1], curves["exact"]["valid_mlogloss"][-1])
    for mode in ("diagonal", "exact"):
        curve = curves[mode]["valid_mlogloss"]
        hit = next((i + 1 for i, v in enumerate(curve) if v <= target), None)
        print(f"  {mode:<9} reaches shared target {target:.5f} at round "
              f"{hit if hit is not None else 'never'}")
    return curves


# --------------------------------------------------------------------------- 2) K scaling


def section_k_scaling():
    print("\n=== 2) K scaling (num_class = 2, 5, 10) ===")
    for k_cap in (2, 5, 10):
        x, y, k = load(k_classes=k_cap)
        dtr = xgb.DMatrix(x, label=y)
        row = {}
        for mode in ("diagonal", "exact"):
            _, total, _ = timed_train(params_for(mode, k), dtr, rounds=20)
            row[mode] = total
        ratio = row["exact"] / row["diagonal"] if row["diagonal"] > 0 else float("nan")
        print(f"  K={k:<2} diagonal={row['diagonal']:.3f}s  exact={row['exact']:.3f}s  "
              f"ratio={ratio:.2f}x")


# --------------------------------------------------------------------------- 3) N scaling


def section_n_scaling():
    print("\n=== 3) N scaling (training rows) ===")
    for n_rows in (200, 600, 1200):
        x, y, k = load(n_rows=n_rows)
        dtr = xgb.DMatrix(x, label=y)
        row = {}
        for mode in ("diagonal", "exact"):
            _, total, _ = timed_train(params_for(mode, k), dtr, rounds=20)
            row[mode] = total
        ratio = row["exact"] / row["diagonal"] if row["diagonal"] > 0 else float("nan")
        print(f"  N={n_rows:<5} diagonal={row['diagonal']:.3f}s  exact={row['exact']:.3f}s  "
              f"ratio={ratio:.2f}x")


# --------------------------------------------------------------------------- 4) max_bin scaling


def section_max_bin_scaling():
    print("\n=== 4) max_bin scaling ===")
    x, y, k = load()
    dtr = xgb.DMatrix(x, label=y)
    for max_bin in (32, 128, 256):
        row = {}
        for mode in ("diagonal", "exact"):
            _, total, _ = timed_train(params_for(mode, k, max_bin=max_bin), dtr, rounds=20)
            row[mode] = total
        ratio = row["exact"] / row["diagonal"] if row["diagonal"] > 0 else float("nan")
        print(f"  max_bin={max_bin:<4} diagonal={row['diagonal']:.3f}s  exact={row['exact']:.3f}s  "
              f"ratio={ratio:.2f}x")


# --------------------------------------------------------------------------- 5) thread scaling


def section_thread_scaling():
    print("\n=== 5) thread scaling (most relevant to Phase 3C) ===")
    x, y, k = load()
    dtr = xgb.DMatrix(x, label=y)
    for nthread in (1, 2, 4):
        row = {}
        for mode in ("diagonal", "exact"):
            _, total, _ = timed_train(params_for(mode, k, nthread=nthread), dtr, rounds=20)
            row[mode] = total
        ratio = row["exact"] / row["diagonal"] if row["diagonal"] > 0 else float("nan")
        print(f"  nthread={nthread:<2} diagonal={row['diagonal']:.3f}s  exact={row['exact']:.3f}s  "
              f"ratio={ratio:.2f}x")
    print("  (ratio shrinking or flat as threads increase is the Phase 3C signal; a ratio that")
    print("   grows with thread count would mean the zero/reduce steps are not actually")
    print("   benefiting from parallelism.)")


# --------------------------------------------------------------------------- 6) peak memory


def section_peak_memory():
    print("\n=== 6) peak memory, illustrated via Phase 3B's cache budget ===")
    print("  Windows' peak working-set counter (monotonic over the process's life), measured")
    print("  as the delta it grows by during each training call -- a coarse proxy for the")
    print("  histogram cache's own footprint, since it includes everything else the process")
    print("  touches too, but immune to a GC pass hiding an already-passed peak.")
    # Deliberately wide (many features -> many bins) and deep, so the O(K^2)-per-bin exact
    # cache has enough nodes*bins to produce a measurable peak over Python/numpy's own baseline.
    from sklearn.datasets import make_classification
    x, y = make_classification(n_samples=4000, n_features=200, n_informative=40, n_classes=8,
                               n_clusters_per_class=2, random_state=0)
    x, y = x.astype(np.float32), y.astype(np.int32)
    k = 8
    dtr = xgb.DMatrix(x, label=y)
    for label, budget in (("tiny cache (max_cached_hist_node=4)", 4),
                         ("effectively unbounded (max_cached_hist_node=65536)", 65536)):
        before = peak_wset_mb()
        timed_train(params_for("exact", k, max_cached_hist_node=budget, max_bin=256,
                               max_depth=8), dtr, rounds=10)
        after = peak_wset_mb()
        print(f"  {label:<52} peak_wset before={before:.1f}MB after={after:.1f}MB "
              f"grew_by={after - before:.1f}MB")


def main():
    print(f"xgboost: {xgb.__file__}")
    print(f"build_info: {xgb.build_info()}")
    results = {}
    results["loss_curves"] = section_loss_curves()
    section_k_scaling()
    section_n_scaling()
    section_max_bin_scaling()
    section_thread_scaling()
    section_peak_memory()
    with open("research/benchmark_phase3d_evidence_raw.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    print("\nraw loss curves written to research/benchmark_phase3d_evidence_raw.json")


if __name__ == "__main__":
    main()
