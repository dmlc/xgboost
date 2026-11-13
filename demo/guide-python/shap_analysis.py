"""Shared SHAP analysis for the CSV training examples."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def run(task):
    try:
        import xgboost as xgb
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        raise SystemExit(f"SHAP analysis requires xgboost, shap and matplotlib: {exc}") from exc

    config_path = HERE / "basic_walkthrough_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        features = config["feature_columns"]
        csv_path = HERE / config["test_csv"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Invalid configuration {config_path}: {exc}") from exc
    if not isinstance(features, list) or not features or not all(isinstance(f, str) for f in features) or len(set(features)) != len(features):
        raise SystemExit("feature_columns must contain unique feature names")
    model_path = HERE / f"{task}_model.json"
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}; run training for {task} first")
    try:
        model = xgb.Booster(model_file=str(model_path))
        model.set_param({"device": "cpu"})
    except (xgb.core.XGBoostError, ValueError) as exc:
        raise SystemExit(f"Could not load model {model_path}: {exc}") from exc
    if model.feature_names != features:
        raise SystemExit("Configured feature names and order do not match the saved model")
    objective = json.loads(model.save_config())["learner"]["objective"]["name"]
    if (task == "regression" and objective != "reg:squarederror") or (task == "classification" and objective not in ("binary:logistic", "multi:softprob")):
        raise SystemExit(f"Unexpected objective {objective!r} for {task}")
    try:
        frame = pd.read_csv(csv_path)
        if frame.empty or not set(features).issubset(frame.columns):
            raise ValueError("CSV must contain rows and all configured features")
        X = frame.loc[:, features].apply(pd.to_numeric, errors="raise")
        if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
            raise ValueError("features must contain finite numeric values")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Could not read features from {csv_path}: {exc}") from exc
    data = xgb.DMatrix(X)
    contrib = model.predict(data, pred_contribs=True, strict_shape=True)
    if contrib.ndim != 3 or contrib.shape[0] != len(X) or contrib.shape[2] != len(features) + 1:
        raise SystemExit(f"Unexpected SHAP output shape: {contrib.shape}")

    groups = contrib.shape[1]
    if task == "regression":
        if groups != 1:
            raise SystemExit("Expected one regression output")
        outputs = [(0, None, HERE / "shap_regression")]
    else:
        classes_path = HERE / "classification_classes.json"
        try:
            classes = json.loads(classes_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Could not read class mapping {classes_path}: {exc}") from exc
        expected = 2 if objective == "binary:logistic" else groups
        if not isinstance(classes, list) or len(classes) != expected or (objective == "binary:logistic" and groups != 1) or (objective == "multi:softprob" and groups < 3):
            raise SystemExit("Class mapping does not match the model outputs")
        outputs = [(0, classes[1], HERE / "shap_classification" / "class_1")] if groups == 1 else [
            (i, label, HERE / "shap_classification" / f"class_{i}") for i, label in enumerate(classes)
        ]

    for index, label, directory in outputs:
        directory.mkdir(parents=True, exist_ok=True)
        values = contrib[:, index, :-1]
        rows = pd.DataFrame(values, columns=features)
        rows.insert(0, "base_value", contrib[:, index, -1])
        rows.insert(0, "row_index", np.arange(len(X)))
        rows.to_csv(directory / "shap_values.csv", index=False)
        importance = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(values).mean(axis=0)})
        importance.sort_values("mean_abs_shap", ascending=False).to_csv(directory / "feature_importance.csv", index=False)
        if groups == 1 or task == "regression":
            for kind, filename in (("bar", "importance.png"), ("dot", "summary.png")):
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.sca(ax)
                shap.summary_plot(values, X, plot_type=kind, show=False, max_display=len(features), sort=False, color_bar=(kind == "dot"), plot_size=None)
                fig.tight_layout()
                fig.savefig(directory / filename, dpi=200, bbox_inches="tight")
                plt.close(fig)
        metadata = {"task": task, "class_index": (1 if groups == 1 and task == "classification" else index) if label is not None else None,
                    "class_label": label, "output": "raw_margin", "row_index": "zero-based data row in test CSV"}
        (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SHAP results: {directory}")

    if task == "classification" and groups > 1:
        panel_count = len(outputs)
        columns = min(3, panel_count)
        rows_count = (panel_count + columns - 1) // columns
        for kind, filename in (("bar", "importance_all_classes.png"), ("dot", "summary_all_classes.png")):
            fig, axes = plt.subplots(rows_count, columns, figsize=(8 * columns, 6 * rows_count), squeeze=False)
            for i, (index, label, _) in enumerate(outputs):
                ax = axes.flat[i]
                plt.sca(ax)
                shap.summary_plot(
                    contrib[:, index, :-1], X, plot_type=kind, show=False,
                    max_display=len(features), sort=False, color_bar=(kind == "dot"),
                    plot_size=None,
                )
                ax.set_title(f"Class: {label}")
            for ax in axes.flat[panel_count:]:
                ax.set_visible(False)
            fig.tight_layout()
            fig.savefig(HERE / "shap_classification" / filename, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"SHAP results: {HERE / 'shap_classification' / filename}")
