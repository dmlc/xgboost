"""
QuadratureSHAP benchmark harness
================================

This script benchmarks CPU ``quadratureshap`` against ``treeshap`` on:

- real datasets from scikit-learn
- an easy synthetic binary task
- harder synthetic noisy tasks

It emits JSON results and two charts:

- accuracy convergence vs quadrature point count
- runtime speedup vs TreeSHAP

Example
-------

Run from the repository root with the local package and library on the path:

  LD_LIBRARY_PATH=$PWD/lib \
  PYTHONPATH=$PWD/python-package \
  python demo/guide-python/quadratureshap_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Workload:
    """A benchmark workload definition."""

    name: str
    family: str
    objective: str
    rounds: int
    num_class: int | None
    build: Callable[
        [np.random.Generator], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]


DEFAULT_POINTS = [4, 6, 8, 10, 12, 16, 20]
DEFAULT_DEPTHS = [4, 8, 16, 30]
DEFAULT_SEED = 20260320
DEFAULT_THREADS = 35
DEFAULT_TEST_ROWS = 512
DEFAULT_RUNS = 3


def _split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    cap_rows: int,
    *,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y if stratify else None,
    )
    if X_test.shape[0] > cap_rows:
        X_test = X_test[:cap_rows]
        y_test = y_test[:cap_rows]
    return X_train, X_test, y_train, y_test


def make_real_workloads(seed: int, cap_rows: int) -> list[Workload]:
    """Build benchmark specs for real scikit-learn datasets."""

    def breast(_: np.random.Generator):
        data = load_breast_cancer()
        return _split_dataset(
            data.data.astype(np.float32),
            data.target.astype(np.float32),
            seed,
            cap_rows,
            stratify=True,
        )

    def diabetes(_: np.random.Generator):
        data = load_diabetes()
        return _split_dataset(
            data.data.astype(np.float32),
            data.target.astype(np.float32),
            seed,
            cap_rows,
            stratify=False,
        )

    def digits(_: np.random.Generator):
        data = load_digits()
        return _split_dataset(
            data.data.astype(np.float32),
            data.target.astype(np.int32),
            seed,
            cap_rows,
            stratify=True,
        )

    return [
        Workload("breast_cancer", "real", "binary:logistic", 200, None, breast),
        Workload("diabetes", "real", "reg:squarederror", 300, None, diabetes),
        Workload("digits", "real", "multi:softprob", 250, 10, digits),
    ]


def make_synthetic_workloads(cap_rows: int) -> list[Workload]:
    """Build benchmark specs for synthetic workloads."""

    def easy_linear(rng: np.random.Generator):
        X = rng.standard_normal((40000, 50), dtype=np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] > 0).astype(np.float32)
        X_test = rng.standard_normal((cap_rows, 50), dtype=np.float32)
        y_test = (X_test[:, 0] + 0.5 * X_test[:, 1] - 0.25 * X_test[:, 2] > 0).astype(
            np.float32
        )
        return X, X_test, y, y_test

    def random_labels(rng: np.random.Generator):
        X = rng.standard_normal((40000, 50), dtype=np.float32)
        y = rng.integers(0, 2, size=40000, dtype=np.int32).astype(np.float32)
        X_test = rng.standard_normal((cap_rows, 50), dtype=np.float32)
        y_test = rng.integers(0, 2, size=cap_rows, dtype=np.int32).astype(np.float32)
        return X, X_test, y, y_test

    def random_regression(rng: np.random.Generator):
        X = rng.standard_normal((40000, 50), dtype=np.float32)
        y = rng.standard_normal(40000).astype(np.float32)
        X_test = rng.standard_normal((cap_rows, 50), dtype=np.float32)
        y_test = rng.standard_normal(cap_rows).astype(np.float32)
        return X, X_test, y, y_test

    return [
        Workload("easy_linear", "synthetic", "binary:logistic", 200, None, easy_linear),
        Workload(
            "random_labels", "synthetic", "binary:logistic", 200, None, random_labels
        ),
        Workload(
            "random_regression",
            "synthetic",
            "reg:squarederror",
            200,
            None,
            random_regression,
        ),
    ]


def margin_shape(predt: np.ndarray) -> np.ndarray:
    """Normalize output-margin predictions into a comparable array shape."""

    predt = np.asarray(predt)
    if predt.ndim == 1:
        return predt
    return predt


def contrib_sum(predt: np.ndarray) -> np.ndarray:
    """Sum SHAP contributions across the feature axis."""

    predt = np.asarray(predt)
    return predt.sum(axis=-1)


def tree_stats(bst: xgb.Booster) -> dict[str, float]:
    """Collect simple structural statistics from a booster dump."""

    dump = bst.get_dump(dump_format="json", with_stats=True)

    def walk(node: dict, depth: int = 0) -> tuple[int, int, int]:
        children = node.get("children", [])
        if not children:
            return depth, 1, 1
        max_depth = depth
        node_count = 1
        leaf_count = 0
        for child in children:
            child_depth, child_nodes, child_leaves = walk(child, depth + 1)
            max_depth = max(max_depth, child_depth)
            node_count += child_nodes
            leaf_count += child_leaves
        return max_depth, node_count, leaf_count

    max_depths = []
    node_counts = []
    leaf_counts = []
    for tree_json in dump:
        tree = json.loads(tree_json)
        max_depth, nodes, leaves = walk(tree)
        max_depths.append(max_depth)
        node_counts.append(nodes)
        leaf_counts.append(leaves)

    return {
        "mean_max_depth": statistics.mean(max_depths),
        "max_max_depth": max(max_depths),
        "mean_nodes": statistics.mean(node_counts),
        "mean_leaves": statistics.mean(leaf_counts),
    }


def evaluate_model(
    bst: xgb.Booster, dtest: xgb.DMatrix, n_points: int, runs: int
) -> dict[str, float]:
    """Measure accuracy and runtime for one quadrature point count."""

    bst.set_param({"shap_algorithm": "treeshap"})
    margin = margin_shape(bst.predict(dtest, output_margin=True))
    treeshap = np.asarray(bst.predict(dtest, pred_contribs=True))

    bst.set_param(
        {"shap_algorithm": "quadratureshap", "quadratureshap_points": str(n_points)}
    )
    quadrature = np.asarray(bst.predict(dtest, pred_contribs=True))

    diff = np.abs(treeshap - quadrature)
    add = np.abs(contrib_sum(quadrature) - margin)

    quadrature_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        bst.predict(dtest, pred_contribs=True)
        quadrature_times.append(time.perf_counter() - t0)

    bst.set_param({"shap_algorithm": "treeshap"})
    treeshap_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        bst.predict(dtest, pred_contribs=True)
        treeshap_times.append(time.perf_counter() - t0)

    quad_mean = statistics.mean(quadrature_times)
    tree_mean = statistics.mean(treeshap_times)
    return {
        "points": n_points,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_additivity_err": float(add.max()),
        "mean_additivity_err": float(add.mean()),
        "quadrature_mean_s": quad_mean,
        "treeshap_mean_s": tree_mean,
        "speedup_vs_treeshap": tree_mean / quad_mean,
    }


# pylint: disable=too-many-arguments,too-many-positional-arguments
def train_model(
    workload: Workload,
    X_train: np.ndarray,
    y_train: np.ndarray,
    depth: int,
    threads: int,
    seed: int,
) -> xgb.Booster:
    """Train one benchmark model for a workload/depth pair."""
    params: dict[str, int | float | str] = {
        "objective": workload.objective,
        "tree_method": "hist",
        "max_depth": depth,
        "eta": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 0.0,
        "seed": seed,
        "nthread": threads,
    }
    if workload.num_class is not None:
        params["num_class"] = workload.num_class

    dtrain = xgb.DMatrix(X_train, label=y_train)
    return xgb.train(params, dtrain, num_boost_round=workload.rounds)


# pylint: disable=too-many-arguments,too-many-locals
def run_benchmarks(
    workloads: list[Workload],
    *,
    points: list[int],
    depths: list[int],
    seed: int,
    threads: int,
    runs: int,
) -> list[dict]:
    """Run the full benchmark sweep and emit row-wise JSON records."""

    rng = np.random.default_rng(seed)
    results: list[dict] = []

    for workload in workloads:
        X_train, X_test, y_train, y_test = workload.build(rng)
        dtest = xgb.DMatrix(X_test, label=y_test)

        for depth in depths:
            bst = train_model(workload, X_train, y_train, depth, threads, seed)
            stats = tree_stats(bst)
            for n_points in points:
                row = {
                    "dataset": workload.name,
                    "family": workload.family,
                    "depth": depth,
                    **stats,
                    **evaluate_model(bst, dtest, n_points, runs),
                }
                results.append(row)
                print(json.dumps(row), flush=True)

    return results


def plot_results(results: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """Create summary plots for accuracy and speed trends."""

    families = ["real", "synthetic"]
    acc_path = out_dir / "quadratureshap_benchmark_accuracy.png"
    speed_path = out_dir / "quadratureshap_benchmark_speedup.png"

    fig, axes = plt.subplots(
        1, len(families), figsize=(12, 4.5), constrained_layout=True
    )
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        rows = [r for r in results if r["family"] == family]
        for name in sorted({r["dataset"] for r in rows}):
            group = [r for r in rows if r["dataset"] == name]
            best = {}
            for points in sorted({r["points"] for r in group}):
                pts = [r["max_abs_diff"] for r in group if r["points"] == points]
                best[points] = max(pts)
            ax.plot(
                list(best.keys()),
                list(best.values()),
                marker="o",
                linewidth=2,
                label=name,
            )
        ax.axhline(1e-5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"{family.title()} workloads")
        ax.set_xlabel("Quadrature points")
        ax.set_ylabel("Worst-case max abs diff")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="upper right")
    fig.suptitle("QuadratureSHAP accuracy convergence")
    fig.savefig(acc_path, dpi=180)

    fig, axes = plt.subplots(
        1, len(families), figsize=(12, 4.5), constrained_layout=True
    )
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        rows = [r for r in results if r["family"] == family]
        for name in sorted({r["dataset"] for r in rows}):
            group = [r for r in rows if r["dataset"] == name]
            best = {}
            for points in sorted({r["points"] for r in group}):
                pts = [r["speedup_vs_treeshap"] for r in group if r["points"] == points]
                best[points] = statistics.mean(pts)
            ax.plot(
                list(best.keys()),
                list(best.values()),
                marker="o",
                linewidth=2,
                label=name,
            )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"{family.title()} workloads")
        ax.set_xlabel("Quadrature points")
        ax.set_ylabel("Mean speedup vs TreeSHAP")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="upper right")
    fig.suptitle("QuadratureSHAP runtime trend")
    fig.savefig(speed_path, dpi=180)

    return acc_path, speed_path


def print_summary(results: list[dict], target_error: float) -> None:
    """Print a compact per-dataset summary table."""

    print("\nSUMMARY")
    print(
        "dataset              family     depth  points  max_diff     add_err      "
        "speedup  mean_nodes"
    )
    for name in sorted({r["dataset"] for r in results}):
        rows = [r for r in results if r["dataset"] == name]
        safe_rows = [
            r
            for r in rows
            if r["max_abs_diff"] <= target_error
            and r["max_additivity_err"] <= target_error
        ]
        if safe_rows:
            best = min(
                safe_rows, key=lambda r: (r["points"], -r["speedup_vs_treeshap"])
            )
        else:
            best = min(rows, key=lambda r: (r["max_abs_diff"], r["points"]))
        print(
            f"{best['dataset']:<20} {best['family']:<9} {best['depth']:<5} "
            f"{best['points']:<6} {best['max_abs_diff']:<12.3e} "
            f"{best['max_additivity_err']:<12.3e} {best['speedup_vs_treeshap']:<7.2f} "
            f"{best['mean_nodes']:<10.1f}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the benchmark harness."""

    parser = argparse.ArgumentParser(
        description="Benchmark QuadratureSHAP against TreeSHAP."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--points", type=int, nargs="+", default=DEFAULT_POINTS)
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=["real", "synthetic", "all"],
        default=["all"],
        help="Which workload families to run.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path.cwd())
    parser.add_argument("--target-error", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    """Entry point for the benchmark harness."""

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    workloads: list[Workload] = []
    selected = set(args.workloads)
    if "all" in selected or "real" in selected:
        workloads.extend(make_real_workloads(args.seed, args.test_rows))
    if "all" in selected or "synthetic" in selected:
        workloads.extend(make_synthetic_workloads(args.test_rows))

    results = run_benchmarks(
        workloads,
        points=args.points,
        depths=args.depths,
        seed=args.seed,
        threads=args.threads,
        runs=args.runs,
    )

    out_json = args.out_dir / "quadratureshap_benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    acc_path, speed_path = plot_results(results, args.out_dir)
    print_summary(results, args.target_error)
    print(f"\nSAVED_JSON={out_json}")
    print(f"SAVED_ACC={acc_path}")
    print(f"SAVED_SPEED={speed_path}")


if __name__ == "__main__":
    main()
