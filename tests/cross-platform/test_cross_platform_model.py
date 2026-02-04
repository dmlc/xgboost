"""Cross-platform model test: Train on GPU (Linux), test inference on macOS."""

import argparse
import sys
from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

import xgboost as xgb

SEED = 2026


def get_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate reproducible synthetic classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=SEED,
    )
    return X.astype(np.float32), y.astype(np.int32)


def train_model(model_path: str) -> None:
    """Train a classification model using GPU and save it."""
    X, y = get_data()

    clf = xgb.XGBClassifier(
        device="cuda",
        n_estimators=50,
        max_depth=6,
        learning_rate=0.3,
        random_state=SEED,
    )
    clf.fit(X, y)

    accuracy = accuracy_score(y, clf.predict(X))

    clf.get_booster().set_attr(expected_accuracy=str(accuracy))
    clf.save_model(model_path)


def test_inference(model_path: str) -> None:
    """Load model, run inference and verify accuracy matches."""
    X, y = get_data()

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    accuracy = accuracy_score(y, clf.predict(X))
    ea = clf.get_booster().attr("expected_accuracy")
    assert ea is not None
    expected_accuracy = float(ea)

    np.testing.assert_allclose(accuracy, expected_accuracy)


def main() -> int:
    """Entry for both training and inference."""
    parser = argparse.ArgumentParser(description="Cross-platform XGBoost model test.")
    parser.add_argument("--train", action="store_true", help="Train model using GPU")
    parser.add_argument("--inference", action="store_true", help="Test inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="cross_platform_model.ubj",
        help="Path to model file",
    )

    args = parser.parse_args()

    if args.train == args.inference:
        print("Error: Specify exactly one of --train or --inference", file=sys.stderr)
        return 1

    if args.train:
        train_model(args.model_path)
    else:
        test_inference(args.model_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
