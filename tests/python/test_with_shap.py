import numpy as np
import pytest

import xgboost as xgb
from xgboost.testing.data import get_california_housing

try:
    import shap
except Exception:
    shap = None
    pass


pytestmark = pytest.mark.skipif(shap is None, reason="Requires shap package")


# xgboost removed ntree_limit in 2.0, which breaks the SHAP package.
@pytest.mark.xfail
def test_with_shap() -> None:
    X, y = get_california_housing()
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
