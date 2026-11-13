"""Plot mean SHAP contributions for grid IDs selected in the configuration."""

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
        raise SystemExit(f"Install xgboost and matplotlib to plot SHAP waterfalls: {exc}") from exc

    try:
        config = json.loads((HERE / "basic_walkthrough_config.json").read_text(encoding="utf-8"))
        task = config["active_task"]
        ids = config["waterfall"]["grid_ids"]
        features = config["feature_columns"]
        if task not in ("classification", "regression"):
            raise ValueError("active_task must be classification or regression")
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("waterfall.grid_ids must be a nonempty list of unique strings")
        if not isinstance(features, list) or not features or any(not isinstance(f, str) or not f for f in features) or len(features) != len(set(features)):
            raise ValueError("feature_columns must contain unique feature names")
        if set(features) & {"grid_id", "longitude", "latitude"}:
            raise ValueError("grid_id and coordinates cannot be used as waterfall feature columns")
        frame = pd.read_csv(HERE / config["test_csv"], dtype={"grid_id": "string"})
        needed = ["grid_id", "longitude", "latitude"] + features
        if frame.empty or not set(needed).issubset(frame.columns):
            raise ValueError(f"test CSV must contain rows and columns: {needed}")
        if frame["grid_id"].isna().any() or frame["grid_id"].str.strip().eq("").any():
            raise ValueError("test CSV contains empty grid_id values")
        duplicates = frame.loc[frame["grid_id"].duplicated(keep=False), "grid_id"].unique().tolist()
        if duplicates:
            raise ValueError(f"duplicate grid_id values in test CSV: {duplicates}")
        missing = sorted(set(ids) - set(frame["grid_id"]))
        if missing:
            raise ValueError(f"grid_ids not found in test CSV: {missing}")
        selected = frame.set_index("grid_id").loc[ids].reset_index()
        numeric = selected[features + ["longitude", "latitude"]].apply(pd.to_numeric, errors="raise")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("selected features and coordinates must be finite numeric values")
        X = numeric[features]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Invalid waterfall input: {exc}") from exc

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
    supported = {"classification": {"binary:logistic", "multi:softprob"}, "regression": {"reg:squarederror"}}
    if objective not in supported[task]:
        raise SystemExit(f"Unexpected model objective for {task}: {objective}")
    data = xgb.DMatrix(X)
    contributions = model.predict(data, pred_contribs=True, strict_shape=True)
    margins = model.predict(data, output_margin=True, strict_shape=True)
    if contributions.ndim != 3 or contributions.shape[0] != len(ids) or contributions.shape[2] != len(features) + 1 or margins.shape != contributions.shape[:2]:
        raise SystemExit("Unexpected SHAP or prediction output shape")
    if not np.allclose(contributions.sum(axis=2), margins, rtol=1e-5, atol=1e-5):
        raise SystemExit("SHAP contributions do not sum to the model raw predictions")

    groups = contributions.shape[1]
    if task == "regression":
        if groups != 1:
            raise SystemExit("Expected one regression output")
        labels = [None]
    else:
        try:
            mapping = json.loads((HERE / "classification_classes.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read classification class mapping: {exc}") from exc
        if not isinstance(mapping, list) or (objective == "binary:logistic" and (groups != 1 or len(mapping) != 2)) or (objective == "multi:softprob" and len(mapping) != groups):
            raise SystemExit("Class mapping does not match model outputs")
        labels = [mapping[1]] if objective == "binary:logistic" else mapping

    output = HERE / f"shap_{task}" / "waterfall"
    output.mkdir(parents=True, exist_ok=True)
    cols = min(3, groups)
    rows = (groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, max(6, len(features) * 0.38) * rows), squeeze=False, layout="constrained")
    records, summaries = [], []
    for group, label in enumerate(labels):
        class_index = (1 if objective == "binary:logistic" else group) if label is not None else None
        mean_values = contributions[:, group, :-1].mean(axis=0)
        base = float(contributions[:, group, -1].mean())
        prediction = float(margins[:, group].mean())
        order = np.argsort(-np.abs(mean_values), kind="stable")
        ordered = mean_values[order]
        starts = base + np.r_[0, np.cumsum(ordered[:-1])]
        ax = axes.flat[group]
        y = np.arange(len(features))
        ax.barh(y, ordered, left=starts, color=["#d73027" if v >= 0 else "#4575b4" for v in ordered])
        ax.set_yticks(y, [features[i] for i in order])
        ax.invert_yaxis()
        ax.axvline(base, color="gray", linestyle="--", linewidth=1, label=f"Base: {base:.4g}")
        ax.axvline(prediction, color="black", linestyle=":", linewidth=1, label=f"Mean output: {prediction:.4g}")
        ax.set_xlabel("Mean model output (raw margin)")
        ax.set_title("Regression" if label is None else f"Class: {label}")
        ax.legend(fontsize="small")
        for i in order:
            records.append({"class_index": class_index, "class_label": label, "feature": features[i], "mean_shap": float(mean_values[i])})
        summaries.append({"class_index": class_index, "class_label": label, "mean_base_value": base, "mean_raw_prediction": prediction})
    for ax in axes.flat[groups:]:
        ax.set_visible(False)
    fig.suptitle(f"Mean SHAP waterfall — {len(ids)} selected grid samples")
    fig.savefig(output / "waterfall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(output / "mean_shap.csv", index=False)
    grid_info = [{"grid_id": grid_id, "longitude": float(lon), "latitude": float(lat)} for grid_id, lon, lat in zip(ids, numeric["longitude"], numeric["latitude"])]
    metadata = {"task": task, "target_column": config[task]["target_column"], "sample_count": len(ids), "selected_grids": grid_info, "output_scale": "raw_margin", "groups": summaries}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Waterfall results: {output}")


if __name__ == "__main__":
    main()
