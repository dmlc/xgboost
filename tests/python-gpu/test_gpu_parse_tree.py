import sys
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
            assert len(x["Categories"]) == 1
