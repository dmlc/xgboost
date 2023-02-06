"""
Demo for prediction using individual trees and model slices
===========================================================
"""
import os

import numpy as np
from scipy.special import logit
from sklearn.datasets import load_svmlight_file

import xgboost as xgb

CURRENT_DIR = os.path.dirname(__file__)
train = os.path.join(CURRENT_DIR, "../data/agaricus.txt.train")
test = os.path.join(CURRENT_DIR, "../data/agaricus.txt.test")


def individual_tree() -> None:
    """Get prediction from each individual tree and combine them together."""
    X_train, y_train = load_svmlight_file(train)
    X_test, y_test = load_svmlight_file(test)
    Xy_train = xgb.QuantileDMatrix(X_train, y_train)

    n_rounds = 4
    # Specify the base score, otherwise xgboost will estimate one from the training
    # data.
    base_score = 0.5
    params = {
        "max_depth": 2,
        "eta": 1,
        "objective": "reg:logistic",
        "tree_method": "hist",
        "base_score": base_score,
    }
    booster = xgb.train(params, Xy_train, num_boost_round=n_rounds)

    # Use logit to inverse the base score back to raw leaf value (margin)
    scores = np.full((X_test.shape[0],), logit(base_score))
    for i in range(n_rounds):
        # - Use output_margin to get raw leaf values
        # - Use iteration_range to get prediction for only one tree
        # - Use previous prediction as base marign for the model
        Xy_test = xgb.DMatrix(X_test, base_margin=scores)

        if i == n_rounds - 1:
            # last round, get the transformed prediction
            scores = booster.predict(
                Xy_test, iteration_range=(i, i + 1), output_margin=False
            )
        else:
            # get raw leaf value for accumulation
            scores = booster.predict(
                Xy_test, iteration_range=(i, i + 1), output_margin=True
            )

    full = booster.predict(xgb.DMatrix(X_test), output_margin=False)
    np.testing.assert_allclose(scores, full)


def model_slices() -> None:
    """Inference with each individual using model slices."""
    X_train, y_train = load_svmlight_file(train)
    X_test, y_test = load_svmlight_file(test)
    Xy_train = xgb.QuantileDMatrix(X_train, y_train)

    n_rounds = 4
    # Specify the base score, otherwise xgboost will estimate one from the training
    # data.
    base_score = 0.5
    params = {
        "max_depth": 2,
        "eta": 1,
        "objective": "reg:logistic",
        "tree_method": "hist",
        "base_score": base_score,
    }
    booster = xgb.train(params, Xy_train, num_boost_round=n_rounds)
    trees = [booster[t] for t in range(n_rounds)]

    # Use logit to inverse the base score back to raw leaf value (margin)
    scores = np.full((X_test.shape[0],), logit(base_score))
    for i, t in enumerate(trees):
        # Feed previous scores into base margin.
        Xy_test = xgb.DMatrix(X_test, base_margin=scores)

        if i == n_rounds - 1:
            # last round, get the transformed prediction
            scores = t.predict(Xy_test, output_margin=False)
        else:
            # get raw leaf value for accumulation
            scores = t.predict(Xy_test, output_margin=True)

    full = booster.predict(xgb.DMatrix(X_test), output_margin=False)
    np.testing.assert_allclose(scores, full)


if __name__ == "__main__":
    individual_tree()
    model_slices()
