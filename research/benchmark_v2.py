"""
Exact vs diagonal multinomial curvature: corrected benchmark protocol.

This replaces ``benchmark_exact_vs_diagonal.py``, whose results were retracted because it
selected hyper-parameter configurations using TEST mlogloss. Everything in that script's
headline output was therefore contaminated by test-set peeking.

DATA SEPARATION CONTRACT
------------------------
TRAIN       model fitting only.
VALIDATION  early stopping, configuration selection, optimisation curves, all targets,
            all fixed-round and fixed-wall-clock comparisons.
TEST        final held-out evaluation only, computed once, AFTER selection is complete.

The contract is enforced structurally, not by discipline: the search phase never constructs
a test DMatrix at all, so no test metric can reach selection even by accident. The test set
is loaded only in the final-evaluation phase, which receives an already-chosen configuration
and cannot change it.

PREDECLARED RULES (fixed before any run; changing them after seeing results invalidates the
comparison)
-----------------------------------------------------------------------------------------
Grid            eta in {0.3, 0.1} x lambda in {1.0, 10.0} x max_depth in {6}
                identical for both modes.
Seeds           0, 1, 2. Split 60/20/20, produced from the seed.
Early stopping  20 rounds without validation improvement.
Selection       For each (dataset, mode): the config minimising the MEAN best-validation
                loss across seeds. One config per dataset and mode, not per seed.
Target          Symmetric and validation-only. The target for each mode is the other mode's
                best mean validation loss. Reachability is reported in both directions; a
                target that is not reached is reported as "not reached", never extrapolated.
Wall-clock      Equal-budget comparison uses each mode's measured per-round elapsed times
                from isolated runs. Budgets are the two modes' own total times.
Budget-bound    Reported per seed. A run that hits the round cap is described as "not
                converged within the benchmark budget" -- never as a bound on its loss.

Usage:
    PYTHONPATH=python-package python research/benchmark_v2.py --rounds 500 --nthread 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time

import numpy as np

import xgboost as xgb

GRID = {"eta": [0.3, 0.1], "lambda": [1.0, 10.0], "max_depth": [6]}
SEEDS = [0, 1, 2]
EARLY_STOPPING = 20
MODES = ("diagonal", "exact")


# --------------------------------------------------------------------------- data


def load(name: str, quick: bool):
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
        x, y = d.data.astype(np.float32), d.target.astype(np.int32)
    else:
        raise ValueError(name)

    cap = {"digits": 10**9, "letter": 10**9, "covertype": 60_000, "mnist": 20_000}[name]
    if quick:
        cap = min(cap, 6_000)
    if len(y) > cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=cap, replace=False)
        x, y = x[idx], y[idx]
    return x, y, int(y.max()) + 1, f"{x.shape[0]}x{x.shape[1]}, K={int(y.max()) + 1}"


def split_indices(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def mlogloss(proba, y):
    p = np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def params_for(mode, cfg, n_classes, nthread):
    return {
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


class RoundTimer(xgb.callback.TrainingCallback):
    """Records cumulative elapsed wall-clock after each boosting round."""

    def __init__(self):
        self.elapsed: list[float] = []
        self._t0 = None

    def before_training(self, model):
        self._t0 = time.perf_counter()
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.elapsed.append(time.perf_counter() - self._t0)
        return False


# ----------------------------------------------------------------- phase A: search


def search_fit(mode, cfg, n_classes, x, y, tr_idx, va_idx, rounds, nthread):
    """Fit on train, evaluate on validation. The TEST SET IS NOT VISIBLE HERE.

    This function deliberately takes only the train and validation index arrays. There is no
    code path by which a test metric could enter the returned record.
    """
    dtr = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
    dva = xgb.DMatrix(x[va_idx], label=y[va_idx])
    timer = RoundTimer()
    history: dict = {}
    t0 = time.perf_counter()
    booster = xgb.train(
        params_for(mode, cfg, n_classes, nthread),
        dtr,
        num_boost_round=rounds,
        evals=[(dtr, "train"), (dva, "valid")],
        evals_result=history,
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
        callbacks=[timer],
    )
    total_time = time.perf_counter() - t0
    valid_curve = [float(v) for v in history["valid"]["mlogloss"]]
    train_curve = [float(v) for v in history["train"]["mlogloss"]]
    rounds_run = len(valid_curve)
    best_round = int(np.argmin(valid_curve)) + 1
    try:
        df = booster.trees_to_dataframe()
        nodes_per_tree = float(len(df)) / max(df["Tree"].nunique(), 1) if len(df) else 0.0
    except Exception:  # noqa: BLE001
        nodes_per_tree = None
    return {
        "rounds_run": rounds_run,
        "best_round": best_round,
        "best_valid": float(min(valid_curve)),
        "train_at_best": train_curve[best_round - 1],
        "valid_curve": valid_curve,
        "train_curve": train_curve,
        "elapsed_curve": timer.elapsed[:rounds_run],
        "total_seconds": total_time,
        "sec_per_round": total_time / max(rounds_run, 1),
        "budget_bound": rounds_run >= rounds,
        "nodes_per_tree": nodes_per_tree,
    }


# ------------------------------------------------------- phase B: final evaluation


def final_fit(mode, cfg, n_classes, x, y, tr_idx, va_idx, te_idx, rounds, nthread):
    """Refit the ALREADY SELECTED configuration and evaluate the test set once."""
    dtr = xgb.DMatrix(x[tr_idx], label=y[tr_idx])
    dva = xgb.DMatrix(x[va_idx], label=y[va_idx])
    booster = xgb.train(
        params_for(mode, cfg, n_classes, nthread),
        dtr,
        num_boost_round=rounds,
        evals=[(dva, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )
    best = booster.best_iteration + 1
    dte = xgb.DMatrix(x[te_idx], label=y[te_idx])
    proba = booster.predict(dte, iteration_range=(0, best))
    return {
        "test_mlogloss": mlogloss(proba, y[te_idx]),
        "test_accuracy": float(np.mean(proba.argmax(axis=1) == y[te_idx])),
        "best_round": best,
    }


# ------------------------------------------------------------------- trace metrics


def valid_at_round(rec, r):
    c = rec["valid_curve"]
    return c[min(r, len(c)) - 1] if c else float("nan")


def valid_within_time(rec, budget):
    """Best validation loss reached within `budget` seconds of training."""
    best = float("inf")
    for v, t in zip(rec["valid_curve"], rec["elapsed_curve"]):
        if t > budget:
            break
        best = min(best, v)
    return best if best < float("inf") else float("nan")


def reach_target(rec, target):
    """(round, seconds) at which validation first reaches `target`, or (None, None)."""
    for i, (v, t) in enumerate(zip(rec["valid_curve"], rec["elapsed_curve"])):
        if v <= target:
            return i + 1, t
    return None, None


# --------------------------------------------------------------------------- main


def run_dataset(name, quick, rounds, nthread, raw_fh):
    x, y, n_classes, desc = load(name, quick)
    print(f"\n=== {name} ({desc}) ===", flush=True)
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]
    splits = {s: split_indices(len(y), s) for s in SEEDS}

    # -- Phase A: search on validation only -----------------------------------
    search: dict = {m: {} for m in MODES}
    for mode in MODES:
        for cfg in configs:
            per_seed = []
            for seed in SEEDS:
                tr_idx, va_idx, _ = splits[seed]
                rec = search_fit(mode, cfg, n_classes, x, y, tr_idx, va_idx, rounds, nthread)
                rec.update({"dataset": name, "n_classes": n_classes, "mode": mode,
                            "seed": seed, "cfg": cfg, "phase": "search"})
                raw_fh.write(json.dumps(rec) + "\n")
                raw_fh.flush()
                per_seed.append(rec)
            search[mode][json.dumps(cfg, sort_keys=True)] = per_seed
            mv = float(np.mean([r["best_valid"] for r in per_seed]))
            print(f"  search {mode:<9} {cfg}  mean_valid={mv:.5f}  "
                  f"rounds={np.mean([r['best_round'] for r in per_seed]):.0f}  "
                  f"{np.mean([r['total_seconds'] for r in per_seed]):.1f}s", flush=True)

    # -- Selection: mean validation across seeds, predeclared -----------------
    selected = {}
    for mode in MODES:
        best_key = min(search[mode],
                       key=lambda k: float(np.mean([r["best_valid"] for r in search[mode][k]])))
        selected[mode] = {"cfg": json.loads(best_key), "records": search[mode][best_key]}
        mv = float(np.mean([r["best_valid"] for r in search[mode][best_key]]))
        print(f"  SELECTED {mode:<9} {selected[mode]['cfg']}  mean_valid={mv:.5f}", flush=True)

    # -- Phase B: test evaluated once, after selection -------------------------
    for mode in MODES:
        finals = []
        for seed in SEEDS:
            tr_idx, va_idx, te_idx = splits[seed]
            fin = final_fit(mode, selected[mode]["cfg"], n_classes, x, y,
                            tr_idx, va_idx, te_idx, rounds, nthread)
            fin.update({"dataset": name, "n_classes": n_classes, "mode": mode,
                        "seed": seed, "cfg": selected[mode]["cfg"], "phase": "final"})
            raw_fh.write(json.dumps(fin) + "\n")
            raw_fh.flush()
            finals.append(fin)
        selected[mode]["finals"] = finals
        print(f"  FINAL    {mode:<9} test={np.mean([f['test_mlogloss'] for f in finals]):.5f}  "
              f"acc={np.mean([f['test_accuracy'] for f in finals]):.4f}", flush=True)

    # -- Symmetric validation targets -----------------------------------------
    mean_valid = {m: float(np.mean([r["best_valid"] for r in selected[m]["records"]]))
                  for m in MODES}
    for mode in MODES:
        other = "exact" if mode == "diagonal" else "diagonal"
        target = mean_valid[other]
        hits = [reach_target(r, target) for r in selected[mode]["records"]]
        ok = [h for h in hits if h[0] is not None]
        if ok:
            print(f"  {mode:<9} reaches {other}'s valid target {target:.5f} in "
                  f"{np.mean([h[0] for h in ok]):.0f} rounds / "
                  f"{np.mean([h[1] for h in ok]):.1f}s  ({len(ok)}/{len(hits)} seeds)",
                  flush=True)
        else:
            print(f"  {mode:<9} does NOT reach {other}'s valid target {target:.5f}", flush=True)
    return {"dataset": name, "n_classes": n_classes, "desc": desc, "selected": selected,
            "mean_valid": mean_valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--nthread", type=int, default=8)
    ap.add_argument("--datasets", default="digits,covertype,mnist,letter")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--raw", default="research/benchmark_v2_raw.jsonl")
    args = ap.parse_args()

    print(f"platform : {platform.platform()}")
    print(f"protocol : grid={GRID} seeds={SEEDS} rounds<={args.rounds} "
          f"early_stopping={EARLY_STOPPING} nthread={args.nthread}")
    print("contract : selection and all targets on VALIDATION; test evaluated once, after "
          "selection")

    with open(args.raw, "w", encoding="utf-8") as fh:
        for name in args.datasets.split(","):
            run_dataset(name.strip(), args.quick, args.rounds, args.nthread, fh)
    print(f"\nraw records: {args.raw}")


if __name__ == "__main__":
    main()
