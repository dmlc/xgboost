import sys
import pytest
import xgboost as xgb

sys.path.append("tests/python")
import testing as tm


def test_tree_to_df_categorical():
    X, y = tm.make_categorical(100, 10, 31, False)
    Xy = xgb.DMatrix(X, y, enable_categorical=True)
    booster = xgb.train({"tree_method": "gpu_hist"}, Xy, num_boost_round=10)
    df = booster.trees_to_dataframe()
    for _, x in df.iterrows():
        if x["Feature"] != "Leaf":
            assert len(x["Category"]) == 1


def test_split_value_histograms():
    X, y = tm.make_categorical(1000, 10, 13, False)
    reg = xgb.XGBRegressor(tree_method="gpu_hist", enable_categorical=True)
    reg.fit(X, y)

    with pytest.raises(ValueError, match="doesn't"):
        reg.get_booster().get_split_value_histogram("3", bins=5)
