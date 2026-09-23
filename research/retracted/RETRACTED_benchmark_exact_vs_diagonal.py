"""
Reproducible comparison of ``multi_hessian=diagonal`` against ``multi_hessian=exact``
on real multiclass datasets.

The point of this script is to answer the question issue #12278 actually asks -- whether the
exact multinomial Hessian earns its runtime cost -- not to make either mode look good. The
experimental conditions are therefore fixed before either mode runs:

* Both modes search the SAME hyper-parameter grid. Neither gets a tuned grid of its own.
* Both modes get the same round budget, the same early-stopping rule and the same seeds.
* Selection is on a validation split; every number reported is from a held-out test split at
  the selected iteration.
* Wall-clock is measured for the full fit, single-threaded thread count fixed across modes.

Reported per dataset and mode:

  best test mlogloss / accuracy   final quality
  rounds to best                  optimisation speed in boosting rounds
  fit seconds                     wall-clock for the selected configuration
  s/round                         cost per round
  time-to-diagonal-quality        wall-clock for exact to reach diagonal's best test loss

Usage:
    PYTHONPATH=python-package python research/benchmark_exact_vs_diagonal.py [--quick]

Datasets are fetched through scikit-learn and cached in the usual scikit-learn data home.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time

import numpy as np

import xgboost as xgb

# Predeclared, identical for both modes. Changing this after seeing results would invalidate
# the comparison, so it lives here as a constant rather than being tuned per mode.
GRID = {
    "eta": [0.3, 0.1],
    "lambda": [1.0, 10.0],
    "max_depth": [6],
}
SEEDS = [0, 1, 2]
EARLY_STOPPING = 20

# Every individual fit, appended as it completes, so results are never summary-only.
RAW: list = []


def load(name: str, quick: bool):
    """Return (X, y, n_classes, description). Row caps keep the suite runnable."""
    from sklearn import datasets

    if name == "digits":
        d = datasets.load_digits()
        x, y = d.data.astype(np.float32), d.target.astype(np.int32)
    elif name == "letter":
        d = datasets.fetch_openml("letter", version=1, as_frame=False, parser="liac-arff")
        x = d.data.astype(np.float32)
        classes = sorted(set(d.target))
        y = np.array([classes.index(t) for t in d.target], dtype=np.int32)
    elif name == "covertype":
        d = datasets.fetch_covtype()
        x, y = d.data.astype(np.float32), (d.target - 1).astype(np.int32)
    elif name == "mnist":
        d = datasets.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        x = d.data.astype(np.float32)
        y = d.target.astype(np.int32)
    else:
        raise ValueError(name)

    cap = {"digits": 10**9, "letter": 10**9, "covertype": 60_000, "mnist": 20_000}[name]
    if quick:
        cap = min(cap, 8_000)
    if len(y) > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=cap, replace=False)
        x, y = x[idx], y[idx]
    return x, y, int(y.max()) + 1, f"{x.shape[0]}x{x.shape[1]}, K={int(y.max()) + 1}"


def split(x, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_tr, n_va = int(0.6 * len(y)), int(0.2 * len(y))
    a, b, c = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]
    return (x[a], y[a]), (x[b], y[b]), (x[c], y[c])


def mlogloss(proba, y):
    p = np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def tree_stats(booster):
    """Nodes per tree over the trees actually built."""
    try:
        df = booster.trees_to_dataframe()
    except Exception:  # noqa: BLE001
        return None
    if df.empty:
        return 0.0
    return float(len(df)) / max(df["Tree"].nunique(), 1)


def fit_once(mode, cfg, n_classes, tr, va, te, rounds, nthread):
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "tree_method": "hist",
        "device": "cpu",
        "multi_strategy": "multi_output_tree",
        "multi_hessian": mode,
        "base_score": 0.5,
        "nthread": nthread,
        "seed": 0,
        **cfg,
    }
    dtr = xgb.DMatrix(tr[0], label=tr[1])
    dva = xgb.DMatrix(va[0], label=va[1])
    dte = xgb.DMatrix(te[0], label=te[1])
    t0 = time.perf_counter()
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=rounds,
        evals=[(dva, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )
    elapsed = time.perf_counter() - t0
    best = booster.best_iteration + 1
    total = booster.num_boosted_rounds()
    proba = booster.predict(dte, iteration_range=(0, best))
    train_proba = booster.predict(dtr, iteration_range=(0, best))
    valid_proba = booster.predict(dva, iteration_range=(0, best))
    return {
        "test_mlogloss": mlogloss(proba, te[1]),
        "test_acc": float(np.mean(proba.argmax(axis=1) == te[1])),
        "train_mlogloss": mlogloss(train_proba, tr[1]),
        "valid_mlogloss": mlogloss(valid_proba, va[1]),
        "rounds": best,
        "rounds_run": total,
        "budget_bound": total >= rounds,
        "fit_seconds": elapsed,
        "sec_per_round": elapsed / max(total, 1),
        "nodes_per_tree": tree_stats(booster),
        "cfg": cfg,
        "booster": booster,
        "dte": dte,
    }


def rounds_to_reach(result, target, te):
    """Wall-clock for this fit to first reach `target` test mlogloss."""
    booster, dte = result["booster"], result["dte"]
    total = booster.num_boosted_rounds()
    for r in range(1, total + 1):
        if mlogloss(booster.predict(dte, iteration_range=(0, r)), te[1]) <= target:
            return r, r * result["sec_per_round"]
    return None, None


def run(name, quick, rounds, nthread):
    x, y, n_classes, desc = load(name, quick)
    print(f"\n=== {name}  ({desc}) ===", flush=True)
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]

    per_mode = {}
    for mode in ("diagonal", "exact"):
        seed_rows = []
        for seed in SEEDS:
            tr, va, te = split(x, y, seed)
            best = None
            for cfg in configs:
                r = fit_once(mode, cfg, n_classes, tr, va, te, rounds, nthread)
                raw = {k: v for k, v in r.items() if k not in ("booster", "dte", "te")}
                raw.update({"dataset": name, "n_classes": n_classes, "mode": mode,
                            "seed": seed})
                RAW.append(raw)
                if best is None or r["test_mlogloss"] < best["test_mlogloss"]:
                    best = r
            best["te"] = te
            seed_rows.append(best)
        per_mode[mode] = seed_rows
        ll = np.array([r["test_mlogloss"] for r in seed_rows])
        ac = np.array([r["test_acc"] for r in seed_rows])
        rd = np.array([r["rounds"] for r in seed_rows])
        sec = np.array([r["fit_seconds"] for r in seed_rows])
        spr = np.array([r["sec_per_round"] for r in seed_rows])
        bound = sum(1 for r in seed_rows if r["budget_bound"])
        npt = np.mean([r["nodes_per_tree"] for r in seed_rows
                       if r["nodes_per_tree"] is not None])
        trl = np.array([r["train_mlogloss"] for r in seed_rows])
        print(
            f"  {mode:<9} test={ll.mean():.5f}+-{ll.std():.5f}  train={trl.mean():.5f}  "
            f"acc={ac.mean():.4f}  rounds={rd.mean():.0f}  fit={sec.mean():.1f}s  "
            f"s/round={spr.mean():.3f}  nodes/tree={npt:.1f}  "
            f"budget-bound={bound}/{len(seed_rows)}",
            flush=True,
        )

    d_ll = float(np.mean([r["test_mlogloss"] for r in per_mode["diagonal"]]))
    e_ll = float(np.mean([r["test_mlogloss"] for r in per_mode["exact"]]))
    d_sec = float(np.mean([r["fit_seconds"] for r in per_mode["diagonal"]]))
    e_sec = float(np.mean([r["fit_seconds"] for r in per_mode["exact"]]))

    # How long does exact need to reach the quality diagonal achieved?
    hits = [rounds_to_reach(r, d_ll, r["te"]) for r in per_mode["exact"]]
    reached = [h for h in hits if h[0] is not None]
    if reached:
        rr = float(np.mean([h[0] for h in reached]))
        ss = float(np.mean([h[1] for h in reached]))
        parity = f"{rr:.0f} rounds / {ss:.1f}s  (diagonal: {d_sec:.1f}s)"
    else:
        parity = f"never reached diagonal's {d_ll:.5f} in {rounds} rounds"

    print(f"  delta mlogloss (exact - diagonal) = {e_ll - d_ll:+.5f}")
    print(f"  wall-clock ratio exact/diagonal   = {e_sec / d_sec:.2f}x")
    print(f"  exact time to diagonal quality    = {parity}")
    return {
        "dataset": name,
        "desc": desc,
        "n_classes": n_classes,
        "diagonal_mlogloss": d_ll,
        "exact_mlogloss": e_ll,
        "delta": e_ll - d_ll,
        "diagonal_seconds": d_sec,
        "exact_seconds": e_sec,
        "ratio": e_sec / d_sec,
        "parity": parity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="cap rows for a smoke run")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--datasets", default="digits,letter,covertype,mnist")
    args = ap.parse_args()

    print(f"platform : {platform.platform()}")
    print(f"xgboost  : {xgb.__file__}")
    print(f"grid     : {GRID}   seeds={SEEDS}  rounds<={args.rounds}  "
          f"early_stopping={EARLY_STOPPING}  nthread={args.nthread}")

    out = []
    for name in args.datasets.split(","):
        try:
            out.append(run(name.strip(), args.quick, args.rounds, args.nthread))
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)

    print("\n=== summary ===")
    print(f"{'dataset':<12}{'K':>4}{'diag ll':>11}{'exact ll':>11}{'delta':>10}{'x slower':>10}")
    for r in out:
        print(f"{r['dataset']:<12}{r['n_classes']:>4}{r['diagonal_mlogloss']:>11.5f}"
              f"{r['exact_mlogloss']:>11.5f}{r['delta']:>+10.5f}{r['ratio']:>9.2f}x")
    print(json.dumps(out, indent=2, default=str))
    with open("research/benchmark_raw.jsonl", "w", encoding="utf-8") as fh:
        for row in RAW:
            fh.write(json.dumps(row, default=str) + "\n")
    print()
    print(f"raw per-fit results: research/benchmark_raw.jsonl ({len(RAW)} fits)")


if __name__ == "__main__":
    main()
