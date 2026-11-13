"""Check predictor multicollinearity across the configured train and test CSVs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "basic_walkthrough_config.json"


def main():
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        features = config["feature_columns"]
        if not isinstance(features, list) or not features or len(features) != len(set(features)):
            raise ValueError("feature_columns must be a nonempty list of unique column names")
        frames = []
        for key in ("train_csv", "test_csv"):
            path = HERE / config[key]
            frame = pd.read_csv(path)
            missing = [column for column in features if column not in frame.columns]
            if frame.empty or missing:
                raise ValueError(f"{path} is empty or missing feature columns: {missing}")
            try:
                predictors = frame[features].apply(pd.to_numeric, errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric predictor in {path}: {exc}") from exc
            if predictors.isna().any().any() or not np.isfinite(predictors.to_numpy()).all():
                raise ValueError(f"Predictors in {path} contain missing or infinite values")
            frames.append(predictors)
    except (OSError, pd.errors.ParserError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Cannot check multicollinearity: {exc}") from exc

    X = pd.concat(frames, ignore_index=True)
    corr = X.corr(method="pearson")
    corr.to_csv(HERE / "multicollinearity_correlation.csv", float_format="%.8g")

    # VIF_j = 1 / (1 - R_j^2), where R_j^2 comes from least squares on
    # centered predictors. The pseudoinverse handles collinear designs.
    centered = X.to_numpy(dtype=float) - X.to_numpy(dtype=float).mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    constant = norms == 0
    normalized = np.zeros_like(centered)
    normalized[:, ~constant] = centered[:, ~constant] / norms[~constant]
    gram_inverse = np.linalg.pinv(normalized.T @ normalized, hermitian=True)
    vifs = []
    for i, name in enumerate(features):
        if constant[i]:
            vif = float("inf")
        else:
            r_squared = float(np.clip(1.0 - gram_inverse[i, i], 0.0, 1.0))
            vif = float("inf") if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)
        vifs.append({"feature": name, "vif": vif})
    vif_table = pd.DataFrame(vifs).sort_values("vif", ascending=False)
    vif_table.to_csv(HERE / "multicollinearity_vif.csv", index=False, float_format="%.8g")

    fig_size = max(8, len(features) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.82))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=90)
    ax.set_yticks(range(len(features)), features)
    ax.set_title("Predictor Pearson Correlation (Train + Test)")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(HERE / "multicollinearity_correlation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    high_pairs = [
        (features[i], features[j], float(corr.iloc[i, j]))
        for i in range(len(features)) for j in range(i + 1, len(features))
        if abs(corr.iloc[i, j]) >= 0.8
    ]
    print(f"Samples: {len(X)} (train + test)")
    print(f"Saved: {HERE / 'multicollinearity_correlation.csv'}")
    print(f"Saved: {HERE / 'multicollinearity_vif.csv'}")
    print(f"Saved: {HERE / 'multicollinearity_correlation.png'}")
    print("VIF >= 10:")
    print(vif_table[vif_table["vif"] >= 10].to_string(index=False) if (vif_table["vif"] >= 10).any() else "None")
    print("|Pearson correlation| >= 0.8 pairs:")
    print("\n".join(f"{a} / {b}: {value:.3f}" for a, b, value in high_pairs) if high_pairs else "None")


if __name__ == "__main__":
    main()
