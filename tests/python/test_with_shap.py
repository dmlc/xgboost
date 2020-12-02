import numpy as np
import xgboost as xgb
import testing as tm
import pytest

try:
    import shap
except ImportError:
    shap = None
    pass

pytestmark = pytest.mark.skipif(shap is None, reason="Requires shap package")


# Check integration is not broken from xgboost side
# Changes in binary format may cause problems
def test_with_shap():
    X, y = shap.datasets.boston()
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train({"learning_rate": 0.01}, dtrain, 10)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    margin = model.predict(dtrain, output_margin=True)
    assert np.allclose(np.sum(shap_values, axis=len(shap_values.shape) - 1),
                       margin - explainer.expected_value, 1e-3, 1e-3)
