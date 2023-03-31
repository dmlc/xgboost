import numpy as np
import pytest

import xgboost as xgb

try:
    import shap
except Exception:
    shap = None
    pass


pytestmark = pytest.mark.skipif(shap is None, reason="Requires shap package")


# xgboost removed ntree_limit in 2.0, which breaks the SHAP package.
@pytest.mark.xfail
def test_with_shap() -> None:
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train({"learning_rate": 0.01}, dtrain, 10)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    margin = model.predict(dtrain, output_margin=True)
    assert np.allclose(
        np.sum(shap_values, axis=len(shap_values.shape) - 1),
        margin - explainer.expected_value,
        1e-3,
        1e-3,
    )
