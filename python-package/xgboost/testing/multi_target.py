"""Tests for multi-target training."""

from typing import Optional

from sklearn.datasets import make_classification, make_multilabel_classification

import xgboost.testing as tm

from ..sklearn import XGBClassifier
from .updater import ResetStrategy
from .utils import Device


def run_multiclass(device: Device, learning_rate: Optional[float]) -> None:
    """Use vector leaf for multi-class models."""
    X, y = make_classification(128, n_features=12, n_informative=10, n_classes=4)
    clf = XGBClassifier(
        multi_strategy="multi_output_tree",
        callbacks=[ResetStrategy()],
        n_estimators=10,
        device=device,
        learning_rate=learning_rate,
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "multi:softprob"
    assert tm.non_increasing(clf.evals_result()["validation_0"]["mlogloss"])

    proba = clf.predict_proba(X)
    assert proba.shape == (y.shape[0], 4)


def run_multilabel(device: Device, learning_rate: Optional[float]) -> None:
    """Use vector leaf for multi-label classification models."""
    X, y = make_multilabel_classification(128)
    clf = XGBClassifier(
        multi_strategy="multi_output_tree",
        callbacks=[ResetStrategy()],
        n_estimators=10,
        device=device,
        learning_rate=learning_rate,
    )
    clf.fit(X, y, eval_set=[(X, y)])
    assert clf.objective == "binary:logistic"
    assert tm.non_increasing(clf.evals_result()["validation_0"]["logloss"])

    proba = clf.predict_proba(X)
    assert proba.shape == y.shape
