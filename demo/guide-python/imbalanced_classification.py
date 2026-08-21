"""
Handling imbalanced classification (credit default style)
============================================================

Real-world tabular classification problems in domains like credit risk and
fraud detection are usually heavily imbalanced: defaults, frauds, or churns
are rare relative to the negative class. Training on such data with default
settings tends to produce a model that's biased toward the majority class.

This example builds a synthetic dataset with roughly a 15-20:1 negative:positive
ratio, similar to a default-prediction problem, and compares a baseline model
against one using ``scale_pos_weight`` to correct for the imbalance.

See Also
--------
- :doc:`param_tuning </tutorials/param_tuning>`

"""

import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def main() -> None:
    X, y = make_classification(
        n_samples=20000,
        n_features=20,
        n_informative=8,
        weights=[0.95],  # ~95% negative: roughly a 15-20:1 imbalance after flip_y noise
        flip_y=0.01,
        random_state=1994,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=1994
    )

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"Train set: {n_neg} negative, {n_pos} positive ({n_neg / n_pos:.1f}:1)")

    # Accuracy is a misleading metric here: predicting "no default" for every
    # row would already score ~95%. Use AUC and average precision (the area
    # under the precision-recall curve) instead, which are far more
    # informative when the positive class is rare.
    baseline = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, eval_metric="aucpr", tree_method="hist"
    )
    baseline.fit(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]

    # The standard rule of thumb: scale_pos_weight = n_negative / n_positive.
    # This multiplies the gradient/Hessian contribution of every positive-class
    # sample by this factor, so misclassifying a rare positive costs the model
    # as much as misclassifying `ratio` negatives would.
    ratio = n_neg / n_pos
    weighted = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=ratio,
    )
    weighted.fit(X_train, y_train)
    weighted_proba = weighted.predict_proba(X_test)[:, 1]

    for name, proba in [
        ("baseline", baseline_proba),
        ("scale_pos_weight", weighted_proba),
    ]:
        auc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        preds = (proba >= 0.5).astype(int)
        _tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(
            f"{name:>17}: AUC={auc:.4f}  avg_precision={ap:.4f}  "
            f"recall@0.5={recall:.3f}  (tp={tp}, fn={fn}, fp={fp})"
        )

    print(
        "\nAUC and average precision are usually close between the two models -- "
        "scale_pos_weight mainly shifts *where* the decision boundary falls, not "
        "the model's ranking ability. The recall/false-positive trade-off at a "
        "fixed 0.5 threshold is where the difference typically shows up: the "
        "weighted model should catch more of the rare positive class, at the "
        "cost of more false positives. In practice, tune the classification "
        "threshold using the validation set's precision-recall curve rather "
        "than relying on the default 0.5 cutoff."
    )


if __name__ == "__main__":
    main()
