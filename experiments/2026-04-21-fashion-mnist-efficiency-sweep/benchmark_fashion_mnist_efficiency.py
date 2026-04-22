"""Run Fashion-MNIST SHAP efficiency-error sweeps."""

from __future__ import annotations

# pylint: disable=missing-function-docstring,too-many-locals
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn import datasets

DEFAULT_DEPTHS = [4, 8, 12, 16, 24, 32, 48, 64]
DEFAULT_POINTS = [4, 6, 8, 16]
DEFAULT_SEED = 20260421
DEFAULT_TEST_ROWS = 512
DEFAULT_THREADS = 35
DEFAULT_ROUNDS = 100
DEFAULT_MAX_LEAVES = 128


def fetch_fashion_mnist() -> tuple[object, np.ndarray]:
    x, y = datasets.fetch_openml("Fashion-MNIST", return_X_y=True)
    return x, y.astype(np.int64)


def tree_stats(model: xgb.Booster) -> dict[str, float]:
    dump = model.get_dump(dump_format="json", with_stats=True)

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

    max_depths: list[int] = []
    node_counts: list[int] = []
    leaf_counts: list[int] = []
    for tree_json in dump:
        tree_json = re.sub(r"\bnan\b", "0", tree_json)
        tree_json = re.sub(r"\binf\b", "0", tree_json)
        tree = json.loads(tree_json)
        max_depth, nodes, leaves = walk(tree)
        max_depths.append(max_depth)
        node_counts.append(nodes)
        leaf_counts.append(leaves)

    return {
        "num_trees": len(dump),
        "mean_max_depth": float(np.mean(max_depths)),
        "max_max_depth": float(np.max(max_depths)),
        "mean_nodes": float(np.mean(node_counts)),
        "mean_leaves": float(np.mean(leaf_counts)),
    }


def train_model(
    x_train: object, y_train: np.ndarray, depth: int, seed: int, max_leaves: int
) -> xgb.Booster:
    dtrain = xgb.QuantileDMatrix(x_train, y_train, enable_categorical=True)
    params: dict[str, object] = {
        "objective": "multi:softmax",
        "num_class": 10,
        "tree_method": "hist",
        "device": "cpu",
        "grow_policy": "lossguide",
        "max_leaves": max_leaves,
        "max_depth": depth,
        "eta": 0.01,
        "seed": seed,
        "nthread": DEFAULT_THREADS,
    }
    return xgb.train(params, dtrain, num_boost_round=DEFAULT_ROUNDS, verbose_eval=False)


def sample_rows(
    x: object, y: np.ndarray, rows: int, seed: int
) -> tuple[object, np.ndarray]:
    rs = np.random.RandomState(seed)
    row_idx = rs.choice(len(y), size=rows, replace=False)
    if hasattr(x, "iloc"):
        return x.iloc[row_idx, :], y[row_idx]
    return x[row_idx, :], y[row_idx]


def efficiency_metrics(pred: np.ndarray, margin: np.ndarray) -> dict[str, float]:
    err = np.abs(np.sum(pred, axis=pred.ndim - 1) - margin).reshape(-1)
    return {
        "mean_efficiency_err": float(np.mean(err)),
        "p99_efficiency_err": float(np.quantile(err, 0.99)),
        "max_efficiency_err": float(np.max(err)),
    }


def predict_contribs(
    booster: xgb.Booster,
    dtest: xgb.DMatrix,
    algorithm: str,
    quadrature_points: int | None,
) -> np.ndarray:
    params: dict[str, object] = {"device": "cpu", "shap_algorithm": algorithm}
    if algorithm == "quadratureshap":
        assert quadrature_points is not None
        params["quadratureshap_points"] = quadrature_points
    booster = booster.copy()
    booster.set_param(params)
    return np.asarray(booster.predict(dtest, pred_contribs=True))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict[str, object]], metric: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    series = {}
    for row in rows:
        series.setdefault(row["algorithm_label"], []).append(row)
    for label, vals in series.items():
        vals = sorted(vals, key=lambda r: r["requested_depth"])
        xs = [r["requested_depth"] for r in vals]
        ys = [r[metric] for r in vals]
        plt.plot(xs, ys, marker="o", linewidth=2, label=label)
    plt.yscale("log")
    plt.xlabel("Requested max_depth")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Fashion-MNIST efficiency sweep: {metric}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    header = (
        "| algorithm | requested_depth | mean_max_depth | max_max_depth | "
        "mean_efficiency_err | p99_efficiency_err | max_efficiency_err |"
    )
    lines = [
        "## Fashion-MNIST Efficiency Sweep",
        "",
        header,
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['algorithm_label']} | {row['requested_depth']} | "
            f"{row['mean_max_depth']:.3f} | {row['max_max_depth']:.0f} | "
            f"{row['mean_efficiency_err']:.6e} | {row['p99_efficiency_err']:.6e} | "
            f"{row['max_efficiency_err']:.6e} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    out_dir: Path, metadata: dict[str, object], rows: list[dict[str, object]]
) -> None:
    (out_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2)
    )
    write_csv(out_dir / "results.csv", rows)
    write_summary(out_dir / "summary.md", rows)
    make_plot(rows, "mean_efficiency_err", out_dir / "efficiency_mean.png")
    make_plot(rows, "p99_efficiency_err", out_dir / "efficiency_p99.png")
    make_plot(rows, "max_efficiency_err", out_dir / "efficiency_max.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--points", type=int, nargs="+", default=DEFAULT_POINTS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-leaves", type=int, default=DEFAULT_MAX_LEAVES)
    parser.add_argument("--reuse-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_json is not None:
        payload = json.loads(args.reuse_json.read_text())
        metadata = payload["metadata"]
        rows = payload["rows"]
        completed_depths = {row["requested_depth"] for row in rows}
        pending_depths = [
            depth for depth in args.depths if depth not in completed_depths
        ]
        if pending_depths:
            x, y = fetch_fashion_mnist()
            x_test, y_test = sample_rows(x, y, args.test_rows, args.seed)
            dtest = xgb.DMatrix(x_test, y_test, enable_categorical=True)
            for depth in pending_depths:
                print(f"Training requested depth {depth}")
                booster = train_model(x, y, depth, args.seed, args.max_leaves)
                stats = tree_stats(booster)
                margin = np.asarray(booster.predict(dtest, output_margin=True))

                treeshap = predict_contribs(booster, dtest, "treeshap", None)
                rows.append(
                    {
                        "algorithm": "treeshap",
                        "algorithm_label": "TreeSHAP",
                        "requested_depth": depth,
                        **stats,
                        **efficiency_metrics(treeshap, margin),
                    }
                )

                for points in args.points:
                    contribs = predict_contribs(
                        booster, dtest, "quadratureshap", points
                    )
                    rows.append(
                        {
                            "algorithm": "quadratureshap",
                            "algorithm_label": f"QuadratureSHAP-{points}",
                            "requested_depth": depth,
                            "quadrature_points": points,
                            **stats,
                            **efficiency_metrics(contribs, margin),
                        }
                    )
                metadata["depths"] = sorted(
                    set(metadata.get("depths", [])) | set(args.depths)
                )
                metadata["points"] = sorted(
                    set(metadata.get("points", [])) | set(args.points)
                )
                metadata["max_leaves"] = args.max_leaves
                write_outputs(args.out_dir, metadata, rows)
    else:
        x, y = fetch_fashion_mnist()
        x_test, y_test = sample_rows(x, y, args.test_rows, args.seed)
        dtest = xgb.DMatrix(x_test, y_test, enable_categorical=True)

        rows = []
        metadata = {
            "seed": args.seed,
            "test_rows": args.test_rows,
            "rounds": DEFAULT_ROUNDS,
            "max_leaves": args.max_leaves,
            "depths": args.depths,
            "points": args.points,
            "threads": DEFAULT_THREADS,
        }
        for depth in args.depths:
            print(f"Training requested depth {depth}")
            booster = train_model(x, y, depth, args.seed, args.max_leaves)
            stats = tree_stats(booster)
            margin = np.asarray(booster.predict(dtest, output_margin=True))

            treeshap = predict_contribs(booster, dtest, "treeshap", None)
            rows.append(
                {
                    "algorithm": "treeshap",
                    "algorithm_label": "TreeSHAP",
                    "requested_depth": depth,
                    **stats,
                    **efficiency_metrics(treeshap, margin),
                }
            )

            for points in args.points:
                contribs = predict_contribs(booster, dtest, "quadratureshap", points)
                rows.append(
                    {
                        "algorithm": "quadratureshap",
                        "algorithm_label": f"QuadratureSHAP-{points}",
                        "requested_depth": depth,
                        "quadrature_points": points,
                        **stats,
                        **efficiency_metrics(contribs, margin),
                    }
                )
            write_outputs(args.out_dir, metadata, rows)

    write_outputs(args.out_dir, metadata, rows)


if __name__ == "__main__":
    main()
