"""Plot mean absolute SHAP interaction values for the selected configured task."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    try:
        import xgboost as xgb
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        raise SystemExit(f"SHAP interaction plots require xgboost and matplotlib: {exc}") from exc

    try:
        config = json.loads((HERE / "basic_walkthrough_config.json").read_text(encoding="utf-8"))
        task = config["active_task"]
        features = config["feature_columns"]
        frame = pd.read_csv(HERE / config["test_csv"])
        if task not in ("classification", "regression"):
            raise ValueError("active_task must be classification or regression")
        if not isinstance(features, list) or not features or len(set(features)) != len(features):
            raise ValueError("feature_columns must be a nonempty list of unique names")
        if frame.empty or not set(features).issubset(frame.columns):
            raise ValueError("test CSV must contain rows and all configured features")
        X = frame.loc[:, features].apply(pd.to_numeric, errors="raise")
        if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
            raise ValueError("features must be finite numeric values")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Invalid interaction SHAP input: {exc}") from exc

    model_path = HERE / f"{task}_model.json"
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}; train {task} first")
    try:
        model = xgb.Booster(model_file=str(model_path))
        model.set_param({"device": "cpu"})
    except (xgb.core.XGBoostError, ValueError) as exc:
        raise SystemExit(f"Could not load model {model_path}: {exc}") from exc
    if model.feature_names != features:
        raise SystemExit("Configured feature names and order do not match the saved model")
    objective = json.loads(model.save_config())["learner"]["objective"]["name"]
    expected_objectives = {"regression": {"reg:squarederror"}, "classification": {"binary:logistic", "multi:softprob"}}
    if objective not in expected_objectives[task]:
        raise SystemExit(f"Model objective {objective!r} does not match selected task {task}")

    interactions = model.predict(xgb.DMatrix(X), pred_interactions=True, strict_shape=True)
    n_features = len(features)
    if interactions.ndim != 4 or interactions.shape[0] != len(X) or interactions.shape[2:] != (n_features + 1, n_features + 1):
        raise SystemExit(f"Unexpected SHAP interaction shape: {interactions.shape}")
    groups = interactions.shape[1]
    if task == "regression":
        if groups != 1:
            raise SystemExit("Regression model must have one output group")
        outputs = [(0, None, HERE / "shap_regression" / "interaction")]
    else:
        try:
            labels = json.loads((HERE / "classification_classes.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read classification_classes.json: {exc}") from exc
        expected = 2 if objective == "binary:logistic" else groups
        if not isinstance(labels, list) or len(labels) != expected:
            raise SystemExit("Class mapping does not match model output groups")
        if objective == "binary:logistic":
            if groups != 1:
                raise SystemExit("Binary model must have one SHAP output group")
            outputs = [(0, labels[1], HERE / "shap_classification" / "interaction" / "class_1")]
        else:
            outputs = [(i, label, HERE / "shap_classification" / "interaction" / f"class_{i}") for i, label in enumerate(labels)]

    feature_labels = features + ["BIAS"]
    matrices = []
    for group, label, output_dir in outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = np.abs(interactions[:, group, :, :]).mean(axis=0)
        matrices.append(matrix)
        result = pd.DataFrame(matrix, index=feature_labels, columns=feature_labels)
        result.to_csv(output_dir / "interaction_mean_abs.csv", float_format="%.10g")
        metadata = {"task": task, "class_label": label, "aggregation": "mean absolute interaction across test samples", "output_scale": "raw margin", "includes_bias": True}
        (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if task == "classification" and len(outputs) > 1:
        count = len(outputs)
        columns = min(3, count)
        rows = (count + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5.5 * rows), squeeze=False)
        vmax = max(float(matrix.max()) for matrix in matrices)
        image = None
        for i, ((_, label, _), matrix) in enumerate(zip(outputs, matrices)):
            ax = axes.flat[i]
            image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(feature_labels)), feature_labels, rotation=90)
            ax.set_yticks(range(len(feature_labels)), feature_labels)
            ax.set_title(f"Class {label}")
        for ax in axes.flat[count:]:
            ax.set_visible(False)
        fig.colorbar(image, ax=list(axes.flat[:count]), label="Mean absolute interaction contribution", shrink=0.85)
        fig.suptitle("Mean absolute SHAP interactions by class (shared scale)")
        fig.tight_layout()
        combined_path = HERE / "shap_classification" / "interaction_heatmap_all_classes.png"
        fig.savefig(combined_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Interaction heatmap: {combined_path}")
    else:
        _, _, output_dir = outputs[0]
        fig, ax = plt.subplots(figsize=(max(9, 0.65 * len(feature_labels)), max(8, 0.6 * len(feature_labels))))
        image = ax.imshow(matrices[0], cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(feature_labels)), feature_labels, rotation=90)
        ax.set_yticks(range(len(feature_labels)), feature_labels)
        ax.set_title("Mean absolute SHAP interaction")
        fig.colorbar(image, ax=ax, label="Mean absolute interaction contribution")
        fig.tight_layout()
        path = output_dir / "interaction_heatmap.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Interaction heatmap: {path}")


if __name__ == "__main__":
    main()
