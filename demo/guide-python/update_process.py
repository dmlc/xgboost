"""
Demo for using `process_type` with `prune` and `refresh`
========================================================

Modifying existing trees is not a well established use for XGBoost, so feel free to
experiment.

"""

import numpy as np
from sklearn.datasets import fetch_california_housing

import xgboost as xgb


def main():
    n_rounds = 32

    X, y = fetch_california_housing(return_X_y=True)

    # Train a model first
    X_train = X[: X.shape[0] // 2]
    y_train = y[: y.shape[0] // 2]
    Xy = xgb.DMatrix(X_train, y_train)
    evals_result: xgb.callback.EvaluationMonitor.EvalsLog = {}
    booster = xgb.train(
        {"tree_method": "gpu_hist", "max_depth": 6},
        Xy,
        num_boost_round=n_rounds,
        evals=[(Xy, "Train")],
        evals_result=evals_result,
    )
    SHAP = booster.predict(Xy, pred_contribs=True)

    # Refresh the leaf value and tree statistic
    X_refresh = X[X.shape[0] // 2:]
    y_refresh = y[y.shape[0] // 2:]
    Xy_refresh = xgb.DMatrix(X_refresh, y_refresh)
    # The model will adapt to other half of the data by changing leaf value (no change in
    # split condition) with refresh_leaf set to True.
    refresh_result: xgb.callback.EvaluationMonitor.EvalsLog = {}
    refreshed = xgb.train(
        {"process_type": "update", "updater": "refresh", "refresh_leaf": True},
        Xy_refresh,
        num_boost_round=n_rounds,
        xgb_model=booster,
        evals=[(Xy, "Original"), (Xy_refresh, "Train")],
        evals_result=refresh_result,
    )

    # Refresh the model without changing the leaf value, but tree statistic including
    # cover and weight are refreshed.
    refresh_result: xgb.callback.EvaluationMonitor.EvalsLog = {}
    refreshed = xgb.train(
        {"process_type": "update", "updater": "refresh", "refresh_leaf": False},
        Xy_refresh,
        num_boost_round=n_rounds,
        xgb_model=booster,
        evals=[(Xy, "Original"), (Xy_refresh, "Train")],
        evals_result=refresh_result,
    )
    # Without refreshing the leaf value, resulting trees should be the same with original
    # model except for accumulated statistic.  The rtol is for floating point error in
    # prediction.
    np.testing.assert_allclose(
        refresh_result["Original"]["rmse"], evals_result["Train"]["rmse"], rtol=1e-5
    )
    # But SHAP value is changed as cover in tree nodes are changed.
    refreshed_SHAP = refreshed.predict(Xy, pred_contribs=True)
    assert not np.allclose(SHAP, refreshed_SHAP, rtol=1e-3)

    # Prune the trees with smaller max_depth
    X_update = X_train
    y_update = y_train
    Xy_update = xgb.DMatrix(X_update, y_update)

    prune_result: xgb.callback.EvaluationMonitor.EvalsLog = {}
    pruned = xgb.train(
        {"process_type": "update", "updater": "prune", "max_depth": 2},
        Xy_update,
        num_boost_round=n_rounds,
        xgb_model=booster,
        evals=[(Xy, "Original"), (Xy_update, "Train")],
        evals_result=prune_result,
    )
    # Have a smaller model, but similar accuracy.
    np.testing.assert_allclose(
        np.array(prune_result["Original"]["rmse"]),
        np.array(prune_result["Train"]["rmse"]),
        atol=1e-5
    )


if __name__ == "__main__":
    main()
