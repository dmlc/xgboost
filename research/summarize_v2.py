"""
Build every benchmark table from ``benchmark_v2_raw.jsonl`` and nothing else.

No number in the output is typed by hand or carried over from a previous run. If a value is
not derivable from the raw records it is printed as "not reached" or "n/a" rather than
estimated.

Usage:
    python research/summarize_v2.py [--raw research/benchmark_v2_raw.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

MODES = ("diagonal", "exact")
ROUND_CHECKPOINTS = (25, 50, 100, 200, 400)


def load(path):
    search, final = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (search if r.get("phase") == "search" else final).append(r)
    return search, final


def selected_cfg(search, dataset, mode):
    """Re-derive the selection from the raw records: mean best-validation across seeds."""
    by_cfg = defaultdict(list)
    for r in search:
        if r["dataset"] == dataset and r["mode"] == mode:
            by_cfg[json.dumps(r["cfg"], sort_keys=True)].append(r)
    if not by_cfg:
        return None, []
    key = min(by_cfg, key=lambda k: float(np.mean([r["best_valid"] for r in by_cfg[k]])))
    return json.loads(key), by_cfg[key]


def valid_at_round(rec, r):
    c = rec["valid_curve"]
    return c[min(r, len(c)) - 1] if c else float("nan")


def valid_within_time(rec, budget):
    best = float("inf")
    for v, t in zip(rec["valid_curve"], rec["elapsed_curve"]):
        if t > budget:
            break
        best = min(best, v)
    return best if best < float("inf") else float("nan")


def reach(rec, target):
    for i, (v, t) in enumerate(zip(rec["valid_curve"], rec["elapsed_curve"])):
        if v <= target:
            return i + 1, t
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="research/benchmark_v2_raw.jsonl")
    args = ap.parse_args()
    search, final = load(args.raw)
    if not search:
        print("no search records found")
        return

    datasets = []
    for r in search:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    sel = {(d, m): selected_cfg(search, d, m) for d in datasets for m in MODES}

    # ---------------------------------------------------------------- TABLE 1
    print("## TABLE 1 - selected configuration (selection on mean validation loss "
          "across seeds)\n")
    print("| Dataset | K | Mode | eta | lambda | depth | Best round | Mean valid loss | "
          "Train@best | Total s | s/round | Nodes/tree |")
    print("|" + "---|" * 12)
    for d in datasets:
        k = next(r["n_classes"] for r in search if r["dataset"] == d)
        for m in MODES:
            cfg, recs = sel[(d, m)]
            if not recs:
                continue
            npt = [r["nodes_per_tree"] for r in recs if r["nodes_per_tree"] is not None]
            print(f"| {d} | {k} | {m} | {cfg['eta']} | {cfg['lambda']} | {cfg['max_depth']} | "
                  f"{np.mean([r['best_round'] for r in recs]):.0f} | "
                  f"{np.mean([r['best_valid'] for r in recs]):.5f} | "
                  f"{np.mean([r['train_at_best'] for r in recs]):.5f} | "
                  f"{np.mean([r['total_seconds'] for r in recs]):.1f} | "
                  f"{np.mean([r['sec_per_round'] for r in recs]):.4f} | "
                  f"{np.mean(npt):.1f} |" if npt else "n/a |")

    # ---------------------------------------------------------------- TABLE 2
    print("\n## TABLE 2 - validation loss at fixed round checkpoints "
          "(mean over seeds, selected config)\n")
    head = "| Dataset | K | Mode | " + " | ".join(f"r={r}" for r in ROUND_CHECKPOINTS) + " |"
    print(head)
    print("|" + "---|" * (3 + len(ROUND_CHECKPOINTS)))
    for d in datasets:
        k = next(r["n_classes"] for r in search if r["dataset"] == d)
        for m in MODES:
            _, recs = sel[(d, m)]
            cells = []
            for cp in ROUND_CHECKPOINTS:
                vals = [valid_at_round(r, cp) for r in recs if len(r["valid_curve"]) >= 1]
                cells.append(f"{np.mean(vals):.5f}" if vals else "n/a")
            print(f"| {d} | {k} | {m} | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- TABLE 3
    print("\n## TABLE 3 - equal wall-clock comparison (best VALIDATION loss within a "
          "shared time budget)\n")
    print("| Dataset | K | Budget (s) | Budget source | diagonal valid | exact valid | "
          "Better at equal compute |")
    print("|" + "---|" * 7)
    for d in datasets:
        k = next(r["n_classes"] for r in search if r["dataset"] == d)
        times = {m: np.mean([r["total_seconds"] for r in sel[(d, m)][1]]) for m in MODES}
        for src, budget in (("diagonal's own total", times["diagonal"]),
                            ("exact's own total", times["exact"])):
            dv = np.mean([valid_within_time(r, budget) for r in sel[(d, "diagonal")][1]])
            ev = np.mean([valid_within_time(r, budget) for r in sel[(d, "exact")][1]])
            if np.isnan(dv) or np.isnan(ev):
                better = "n/a"
            else:
                better = "exact" if ev < dv else ("diagonal" if dv < ev else "tie")
            print(f"| {d} | {k} | {budget:.1f} | {src} | {dv:.5f} | {ev:.5f} | {better} |")

    # ---------------------------------------------------------------- TABLE 3b
    print("\n## TABLE 3b - symmetric validation-target reachability "
          "(target = the OTHER mode's mean best validation loss)\n")
    print("| Dataset | K | Mode | Target (valid) | Reached | Rounds | Seconds | Seeds |")
    print("|" + "---|" * 8)
    for d in datasets:
        k = next(r["n_classes"] for r in search if r["dataset"] == d)
        mv = {m: float(np.mean([r["best_valid"] for r in sel[(d, m)][1]])) for m in MODES}
        for m in MODES:
            other = "exact" if m == "diagonal" else "diagonal"
            target = mv[other]
            hits = [reach(r, target) for r in sel[(d, m)][1]]
            ok = [h for h in hits if h[0] is not None]
            if ok:
                print(f"| {d} | {k} | {m} | {target:.5f} | yes | "
                      f"{np.mean([h[0] for h in ok]):.0f} | "
                      f"{np.mean([h[1] for h in ok]):.1f} | {len(ok)}/{len(hits)} |")
            else:
                print(f"| {d} | {k} | {m} | {target:.5f} | **not reached** | - | - | "
                      f"0/{len(hits)} |")

    # ---------------------------------------------------------------- TABLE 4
    print("\n## TABLE 4 - final held-out TEST result "
          "(evaluated once, after selection)\n")
    print("| Dataset | K | Mode | Test mlogloss | Test accuracy | Best round | Seeds |")
    print("|" + "---|" * 7)
    for d in datasets:
        k = next((r["n_classes"] for r in final if r["dataset"] == d), None)
        for m in MODES:
            f = [r for r in final if r["dataset"] == d and r["mode"] == m]
            if not f:
                print(f"| {d} | {k} | {m} | n/a | n/a | n/a | 0 |")
                continue
            print(f"| {d} | {k} | {m} | "
                  f"{np.mean([r['test_mlogloss'] for r in f]):.5f} | "
                  f"{np.mean([r['test_accuracy'] for r in f]):.4f} | "
                  f"{np.mean([r['best_round'] for r in f]):.0f} | {len(f)} |")

    # ---------------------------------------------------------------- TABLE 5
    print("\n## TABLE 5 - convergence status PER SEED "
          "(selected config; 'not converged' = hit the round cap)\n")
    print("| Dataset | Mode | Seed | Rounds run | Best round | Status |")
    print("|" + "---|" * 6)
    for d in datasets:
        for m in MODES:
            for r in sorted(sel[(d, m)][1], key=lambda q: q["seed"]):
                status = ("not converged within the benchmark budget"
                          if r["budget_bound"] else "converged (early stopped)")
                print(f"| {d} | {m} | {r['seed']} | {r['rounds_run']} | "
                      f"{r['best_round']} | {status} |")

    # -------------------------------------------------------------- APPENDIX
    print("\n## APPENDIX - every search fit\n")
    print("| Dataset | Mode | Seed | eta | lambda | depth | Rounds run | Best round | "
          "Best valid | Train@best | s | Converged |")
    print("|" + "---|" * 12)
    for r in sorted(search, key=lambda q: (datasets.index(q["dataset"]), q["mode"],
                                           str(q["cfg"]), q["seed"])):
        c = r["cfg"]
        print(f"| {r['dataset']} | {r['mode']} | {r['seed']} | {c['eta']} | {c['lambda']} | "
              f"{c['max_depth']} | {r['rounds_run']} | {r['best_round']} | "
              f"{r['best_valid']:.5f} | {r['train_at_best']:.5f} | "
              f"{r['total_seconds']:.1f} | {'no' if r['budget_bound'] else 'yes'} |")

    print(f"\n{len(search)} search fits, {len(final)} final evaluations, "
          f"{len(datasets)} datasets")


if __name__ == "__main__":
    main()
