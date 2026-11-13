"""Regression training template, called by basic_walkthrough.py."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run(train, test, features, target, config, output_dir):
    import xgboost as xgb

    try:
        train_y = pd.to_numeric(train[target], errors="raise")
        test_y = pd.to_numeric(test[target], errors="raise")
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Regression target must be numeric: {exc}") from exc
    params = dict(config["params"])
    params["objective"] = "reg:squarederror"
    params["eval_metric"] = "rmse"
    dtrain = xgb.DMatrix(train[features], label=train_y)
    dtest = xgb.DMatrix(test[features], label=test_y)
    model = xgb.train(params, dtrain, num_boost_round=config["num_boost_round"], evals=[(dtest, "test")])
    predicted = model.predict(dtest)
    model.save_model(str(output_dir / "regression_model.json"))
    metrics = {"task": "regression", "mae": float(mean_absolute_error(test_y, predicted)), "rmse": float(np.sqrt(mean_squared_error(test_y, predicted)))}
    (output_dir / "regression_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
