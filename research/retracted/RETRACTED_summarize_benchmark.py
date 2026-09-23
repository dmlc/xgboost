"""
Turn ``research/benchmark_raw.jsonl`` into the two comparison tables.

Reads only the raw per-fit records, so the tables cannot disagree with the measurements. No
dataset is excluded and no configuration is dropped: the per-dataset row is the configuration
selected on validation, and the appendix prints every fit.

Usage:
    python research/summarize_benchmark.py [--raw research/benchmark_raw.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select(rows):
    """Per (dataset, mode, seed) pick the config with the lowest validation loss."""
    best = {}
    for r in rows:
        key = (r["dataset"], r["mode"], r["seed"])
        if key not in best or r["valid_mlogloss"] < best[key]["valid_mlogloss"]:
            best[key] = r
    return best


def agg(selected, dataset, mode, field):
    vals = [v[field] for k, v in selected.items()
            if k[0] == dataset and k[1] == mode and v[field] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="research/benchmark_raw.jsonl")
    args = ap.parse_args()

    rows = load(args.raw)
    selected = select(rows)
    datasets = []
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    print("## Table 1 - selected configuration per dataset and mode "
          "(mean over seeds; selection on validation, reported on test)\n")
    hdr = ("| Dataset | K | Mode | Rounds | Budget-bound | Train Loss | Val Loss | "
           "Test Loss | Accuracy | Time (s) | s/round | Nodes/Tree |")
    print(hdr)
    print("|" + "---|" * 12)
    for d in datasets:
        k = next(r["n_classes"] for r in rows if r["dataset"] == d)
        for mode in ("diagonal", "exact"):
            bound = sum(1 for kk, v in selected.items()
                        if kk[0] == d and kk[1] == mode and v["budget_bound"])
            n = sum(1 for kk in selected if kk[0] == d and kk[1] == mode)
            print(f"| {d} | {k} | {mode} | {agg(selected, d, mode, 'rounds'):.0f} | "
                  f"{bound}/{n} | {agg(selected, d, mode, 'train_mlogloss'):.5f} | "
                  f"{agg(selected, d, mode, 'valid_mlogloss'):.5f} | "
                  f"{agg(selected, d, mode, 'test_mlogloss'):.5f} | "
                  f"{agg(selected, d, mode, 'test_acc'):.4f} | "
                  f"{agg(selected, d, mode, 'fit_seconds'):.1f} | "
                  f"{agg(selected, d, mode, 'sec_per_round'):.4f} | "
                  f"{agg(selected, d, mode, 'nodes_per_tree'):.1f} |")

    print("\n## Table 2 - the trade-off\n")
    print("| Dataset | K | Exact rounds advantage | Exact runtime cost | "
          "Exact quality difference | Interpretation |")
    print("|" + "---|" * 6)
    for d in datasets:
        k = next(r["n_classes"] for r in rows if r["dataset"] == d)
        dr, er = agg(selected, d, "diagonal", "rounds"), agg(selected, d, "exact", "rounds")
        ds, es = (agg(selected, d, "diagonal", "fit_seconds"),
                  agg(selected, d, "exact", "fit_seconds"))
        dl, el = (agg(selected, d, "diagonal", "test_mlogloss"),
                  agg(selected, d, "exact", "test_mlogloss"))
        d_bound = sum(1 for kk, v in selected.items()
                      if kk[0] == d and kk[1] == "diagonal" and v["budget_bound"])
        rounds_adv = f"{dr / er:.2f}x fewer" if er else "n/a"
        cost = f"{es / ds:.2f}x slower" if ds else "n/a"
        delta = el - dl
        quality = f"{delta:+.5f} ({'exact worse' if delta > 0 else 'exact better'})"
        if d_bound:
            note = "diagonal still budget-bound: its loss is an upper bound, exact's is not"
        elif delta > 0.001:
            note = "exact converges sooner but to a worse point"
        elif delta < -0.001:
            note = "exact converges sooner and to a better point"
        else:
            note = "quality is a wash; only cost differs"
        print(f"| {d} | {k} | {rounds_adv} | {cost} | {quality} | {note} |")

    print("\n## Appendix - every fit\n")
    print("| Dataset | Mode | Seed | eta | lambda | depth | Rounds | Run | Bound | "
          "Train | Val | Test | Acc | s | Nodes/Tree |")
    print("|" + "---|" * 15)
    for r in sorted(rows, key=lambda q: (datasets.index(q["dataset"]), q["mode"],
                                         q["seed"], str(q["cfg"]))):
        c = r["cfg"] if isinstance(r["cfg"], dict) else json.loads(str(r["cfg"]).replace("'", '"'))
        npt = r["nodes_per_tree"]
        print(f"| {r['dataset']} | {r['mode']} | {r['seed']} | {c['eta']} | "
              f"{c['lambda']} | {c['max_depth']} | {r['rounds']} | {r['rounds_run']} | "
              f"{'Y' if r['budget_bound'] else 'n'} | {r['train_mlogloss']:.5f} | "
              f"{r['valid_mlogloss']:.5f} | {r['test_mlogloss']:.5f} | "
              f"{r['test_acc']:.4f} | {r['fit_seconds']:.1f} | "
              f"{npt:.1f} |" if npt is not None else "n/a |")

    print(f"\n{len(rows)} fits, {len(selected)} selected "
          f"({len(datasets)} datasets x 2 modes x seeds)")


if __name__ == "__main__":
    main()
