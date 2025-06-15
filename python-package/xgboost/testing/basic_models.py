"""Tests for basic features of the Booster."""

from typing import Tuple

import numpy as np

from xgboost import testing as tm

from ..core import Booster, DMatrix
from ..training import cv, train
from .utils import Device


def run_custom_objective(  # pylint: disable=too-many-locals
    tree_method: str,
    device: Device,
    dtrain: DMatrix,
    dtest: DMatrix,
) -> None:
    """Tests custom objective and metric functions."""
    param = {
        "max_depth": 2,
        "eta": 1,
        "objective": "reg:logistic",
        "tree_method": tree_method,
        "device": device,
    }
    watchlist = [(dtest, "eval"), (dtrain, "train")]
    num_round = 10

    def evalerror(preds: np.ndarray, dtrain: DMatrix) -> Tuple[str, np.float64]:
        return tm.eval_error_metric(preds, dtrain, rev_link=True)

    # test custom_objective in training
    bst = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        obj=tm.logregobj,
        custom_metric=evalerror,
    )
    assert isinstance(bst, Booster)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    err = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(
        len(preds)
    )
    assert err < 0.1

    # test custom_objective in cross-validation
    cv(
        param,
        dtrain,
        num_round,
        nfold=5,
        seed=0,
        obj=tm.logregobj,
        custom_metric=evalerror,
    )

    # test maximize parameter
    def neg_evalerror(preds: np.ndarray, dtrain: DMatrix) -> Tuple[str, float]:
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))
        return "error", float(sum(labels == (preds > 0.0))) / len(labels)

    bst2 = train(
        param,
        dtrain,
        num_round,
        evals=watchlist,
        obj=tm.logregobj,
        custom_metric=neg_evalerror,
        maximize=True,
    )
    preds2 = bst2.predict(dtest)
    err2 = sum(
        1 for i in range(len(preds2)) if int(preds2[i] > 0.5) != labels[i]
    ) / float(len(preds2))
    assert err == err2
