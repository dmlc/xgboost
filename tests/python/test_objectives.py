import numpy as np
from sklearn.datasets import make_classification

import xgboost as xgb
from xgboost.testing.updater import get_basescore


def test_exp_family() -> None:
    X, y = make_classification(n_samples=128, n_classes=2, weights=[0.8, 0.2])
    clf = xgb.train(
        {"objective": "binary:logistic"}, xgb.QuantileDMatrix(X, y), num_boost_round=1
    )
    reg = xgb.train(
        {"objective": "reg:logistic"}, xgb.QuantileDMatrix(X, y), num_boost_round=1
    )
    clf1 = xgb.train(
        {"objective": "binary:logitraw"}, xgb.QuantileDMatrix(X, y), num_boost_round=1
    )
    # The base score stored in the booster model is un-transformed
    np.testing.assert_allclose([get_basescore(m) for m in (reg, clf, clf1)], y.mean())

    X, y = make_classification(weights=[0.8, 0.2], random_state=2025)
    clf = xgb.train(
        {"objective": "binary:logistic", "scale_pos_weight": 4.0},
        xgb.QuantileDMatrix(X, y),
        num_boost_round=1,
    )
    score = get_basescore(clf)
    np.testing.assert_allclose(score, 0.5, rtol=1e-3)

    _, y1 = make_classification(weights=[0.8, 0.2], random_state=1)
    _, y2 = make_classification(weights=[0.8, 0.2], random_state=3)
    y = np.stack([y, y1, y2], axis=1)
    clf = xgb.train(
        {"objective": "binary:logistic", "scale_pos_weight": 4.0},
        xgb.QuantileDMatrix(X, y),
        num_boost_round=1,
    )
    score = get_basescore(clf)
    np.testing.assert_allclose(score, 0.5, rtol=1e-1)
