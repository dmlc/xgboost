"""
How the cost of ``multi_hessian=exact`` scales with the number of classes.

Theory says histogram storage per bin grows as O(K^2) and each leaf solve as O(K^3). This
measures the end-to-end consequence on one machine, holding everything except ``num_class``
fixed, so the reported ratios are an empirical scalability profile rather than a claim about
any particular workload.

The data is synthetic on purpose: a real dataset changes its own difficulty with K, which
would confound the measurement. Rows, columns, depth and round count are constant across K.

Usage:
    PYTHONPATH=python-package python research/benchmark_k_scaling.py
"""

from __future__ import annotations

import platform
import time

import numpy as np

import xgboost as xgb

ROWS, COLS, ROUNDS, DEPTH = 20_000, 32, 20, 6
K_VALUES = (3, 7, 10, 20, 26)


def make(n_classes, seed=7):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=ROWS)
    x = rng.normal(size=(ROWS, COLS))
    for k in range(n_classes):
        x[y == k, k % COLS] += 1.2
    return x.astype(np.float32), y.astype(np.float32)


def timed(n_classes, mode, nthread):
    x, y = make(n_classes)
    d = xgb.DMatrix(x, label=y)
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "tree_method": "hist",
        "device": "cpu",
        "multi_strategy": "multi_output_tree",
        "multi_hessian": mode,
        "max_depth": DEPTH,
        "eta": 0.3,
        "lambda": 1.0,
        "base_score": 0.5,
        "nthread": nthread,
        "seed": 0,
    }
    xgb.train(params, d, num_boost_round=2)  # warm up
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        xgb.train(params, d, num_boost_round=ROUNDS)
        best = min(best, time.perf_counter() - t0)
    return best


def main(nthread=8):
    print(f"platform : {platform.platform()}")
    print(f"data     : {ROWS} rows x {COLS} cols, {ROUNDS} rounds, max_depth={DEPTH}, "
          f"nthread={nthread}; best of 3 after a warm-up")
    print(f"{'K':>4}{'diagonal s':>13}{'exact s':>11}{'ratio':>9}"
          f"{'exact/K=3':>12}{'bytes/bin':>12}")
    base = None
    for k in K_VALUES:
        d = timed(k, "diagonal", nthread)
        e = timed(k, "exact", nthread)
        if base is None:
            base = e
        # Exact stores (K-1) gradients + (K-1)K/2 Hessian entries per bin, in double.
        per_bin = 8 * ((k - 1) + (k - 1) * k // 2)
        print(f"{k:>4}{d:>13.3f}{e:>11.3f}{e / d:>8.2f}x{e / base:>11.2f}x{per_bin:>12}")
    print()
    print("Storage is exact; the time ratios are one machine, one data shape. They are a")
    print("scalability profile, not a universal benchmark, and not a hard ceiling on K.")


if __name__ == "__main__":
    main()
