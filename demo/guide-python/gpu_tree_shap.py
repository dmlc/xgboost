"""
Use GPU to speedup SHAP value computation
=========================================

Demonstrates using GPU acceleration to compute SHAP values for feature importance.

"""
from urllib.error import HTTPError

import shap
from sklearn.datasets import fetch_california_housing, make_regression

import xgboost as xgb

# Fetch dataset using sklearn
try:
    _data = fetch_california_housing(return_X_y=True)
    X = _data.data
    y = _data.target
    feature_names = _data.feature_names
    print(_data.DESCR)
except HTTPError:
    # Use a synthetic dataset instead if we couldn't
    X, y = make_regression(n_samples=20640, n_features=8, random_state=1234)
    feature_names = [f"f{i}" for i in range(8)]

num_round = 500

param = {
    "eta": 0.05,
    "max_depth": 10,
    "tree_method": "hist",
    "device": "cuda",
}

# GPU accelerated training
dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
model = xgb.train(param, dtrain, num_round)

# Compute shap values using GPU with xgboost
model.set_param({"device": "cuda"})
shap_values = model.predict(dtrain, pred_contribs=True)

# Compute shap interaction values using GPU
shap_interaction_values = model.predict(dtrain, pred_interactions=True)


# shap will call the GPU accelerated version as long as the device parameter is set to
# "cuda"
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X[0, :],
    feature_names=feature_names,
    matplotlib=True,
)

# Show a summary of feature importance
shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)
