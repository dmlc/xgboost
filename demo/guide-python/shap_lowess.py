"""Create LOWESS-smoothed SHAP dependence plots for the configured task."""

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
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except (ImportError, OSError) as exc:
        raise SystemExit(f"Install xgboost, matplotlib, and statsmodels to create LOWESS SHAP plots: {exc}") from exc

    config_path = HERE / "basic_walkthrough_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task = config["active_task"]
        if task not in ("classification", "regression"):
            raise ValueError("active_task must be classification or regression")
        features = config["feature_columns"]
        frame = pd.read_csv(HERE / config["test_csv"])
        if not isinstance(features, list) or not features or len(features) != len(set(features)):
            raise ValueError("feature_columns must be a nonempty list of unique names")
        if frame.empty or not set(features).issubset(frame.columns):
            raise ValueError("test CSV must contain rows and all configured features")
        X = frame.loc[:, features].apply(pd.to_numeric, errors="raise")
        if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
            raise ValueError("features must be finite numeric values")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Invalid LOWESS SHAP input: {exc}") from exc

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
    contrib = model.predict(xgb.DMatrix(X), pred_contribs=True, strict_shape=True)
    if contrib.ndim != 3 or contrib.shape[0] != len(X) or contrib.shape[2] != len(features) + 1:
        raise SystemExit(f"Unexpected SHAP output shape: {contrib.shape}")

    if task == "regression":
        if contrib.shape[1] != 1:
            raise SystemExit("Regression model must have one output group")
        classes = [None]
        output_dir = HERE / "shap_regression" / "lowess_dependence"
    else:
        if objective == "binary:logistic":
            classes_path = HERE / "classification_classes.json"
            try:
                labels = json.loads(classes_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Could not read class mapping {classes_path}: {exc}") from exc
            if not isinstance(labels, list) or len(labels) != 2 or contrib.shape[1] != 1:
                raise SystemExit("Binary model class mapping must contain two labels")
            classes = [labels[1]]  # binary logistic margin is for encoded positive class
        else:
            classes_path = HERE / "classification_classes.json"
            try:
                classes = json.loads(classes_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Could not read class mapping {classes_path}: {exc}") from exc
            if not isinstance(classes, list) or len(classes) != contrib.shape[1]:
                raise SystemExit("Class mapping does not match model output groups")
        output_dir = HERE / "shap_classification" / "lowess_dependence"

    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_index, feature in enumerate(features):
        fig, ax = plt.subplots(figsize=(9, 6))
        csv_data = {"row_index": np.arange(len(X)), "feature_value": X[feature].to_numpy()}
        for class_index, label in enumerate(classes):
            shap_values = contrib[:, class_index, feature_index]
            csv_data[f"shap_class_{class_index}"] = shap_values
            color = f"C{class_index}"
            ax.scatter(X[feature], shap_values, s=18, alpha=0.35, color=color, label=f"Class {label}" if label is not None else None)
            curve = lowess(shap_values, X[feature].to_numpy(), frac=0.6, it=0, return_sorted=True)
            ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=2.2, label=f"LOWESS {label}" if label is not None else "LOWESS")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value (raw model output scale)")
        ax.set_title(f"SHAP dependence with LOWESS: {feature}")
        if task == "classification":
            ax.legend(fontsize="small", ncol=2)
        fig.tight_layout()
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in feature)
        fig.savefig(output_dir / f"{safe_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(csv_data).to_csv(output_dir / f"{safe_name}.csv", index=False)
    print(f"Created {len(features)} feature dependence plots and data tables in {output_dir}")


if __name__ == "__main__":
    main()
