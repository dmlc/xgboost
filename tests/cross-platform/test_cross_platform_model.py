"""Cross-platform model test: Train on GPU (Linux), test inference on macOS."""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

SEED = 2026


def _pickle_path(model_path: str) -> Path:
    return Path(model_path).with_suffix(".pkl")


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
    """Train models using GPU and save them (binary + pickle with column sampling)."""
    X, y = get_data()

    clf = xgb.XGBClassifier(
        device="cuda",
        n_estimators=50,
        max_depth=6,
        learning_rate=0.3,
        random_state=SEED,
        colsample_bynode=0.8,
    )
    clf.fit(X, y)

    accuracy = accuracy_score(y, clf.predict(X))
    clf.get_booster().set_attr(expected_accuracy=str(accuracy))
    clf.save_model(model_path)

    with open(_pickle_path(model_path), "wb") as fd:
        pickle.dump(clf.get_booster(), fd)


def test_inference(model_path: str) -> None:
    """Load models and verify predictions match (binary + pickle)."""
    X, y = get_data()

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    accuracy = accuracy_score(y, clf.predict(X))
    ea = clf.get_booster().attr("expected_accuracy")
    assert ea is not None
    expected_accuracy = float(ea)
    np.testing.assert_allclose(accuracy, expected_accuracy)

    with open(_pickle_path(model_path), "rb") as f:
        booster = pickle.load(f)

    clf = xgb.XGBClassifier(n_estimators=2)
    clf.fit(X, y, xgb_model=booster)


def main() -> int:
    """Entry for both training and inference."""
    parser = argparse.ArgumentParser(description="Cross-platform XGBoost model test.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train models using GPU")
    group.add_argument("--inference", action="store_true", help="Test inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="cross_platform_model.ubj",
        help="Path to model file (pickle path is derived by replacing extension)",
    )

    args = parser.parse_args()

    if args.train:
        train_model(args.model_path)
    else:
        test_inference(args.model_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
