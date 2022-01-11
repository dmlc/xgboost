"""
Getting started with categorical data
=====================================

Experimental support for categorical data.  After 1.5 XGBoost `gpu_hist` tree method has
experimental support for one-hot encoding based tree split, and in 1.6 `approx` supported
was added.

In before, users need to run an encoder themselves before passing the data into XGBoost,
which creates a sparse matrix and potentially increase memory usage.  This demo showcases
the experimental categorical data support, more advanced features are planned.

Also, see :doc:`the tutorial </tutorials/categorical>` for using XGBoost with categorical data


    .. versionadded:: 1.5.0

"""
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple
import argparse


def make_categorical(
    n_samples: int, n_features: int, n_categories: int, onehot: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """Make some random data for demo."""
    rng = np.random.RandomState(1994)

    pd_dict = {}
    for i in range(n_features + 1):
        c = rng.randint(low=0, high=n_categories, size=n_samples)
        pd_dict[str(i)] = pd.Series(c, dtype=np.int64)

    df = pd.DataFrame(pd_dict)
    label = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    for i in range(0, n_features):
        label += df.iloc[:, i]
    label += 1

    df = df.astype("category")
    categories = np.arange(0, n_categories)
    for col in df.columns:
        df[col] = df[col].cat.set_categories(categories)

    if onehot:
        return pd.get_dummies(df), label
    return df, label


def main(args) -> None:
    # Use builtin categorical data support
    # For scikit-learn interface, the input data must be pandas DataFrame or cudf
    # DataFrame with categorical features
    X, y = make_categorical(1024, 16, 7, False)
    # Specify `enable_categorical` to True.
    # tree_method = "approx"
    # tree_method = "gpu_hist"
    tree_method = args.tm
    reg = xgb.XGBRegressor(
        tree_method=tree_method,
        enable_categorical=True,
        max_cat_to_onehot=1,
        n_estimators=8,
        n_jobs=1,
        # max_depth=3,
    )
    reg.fit(X, y, eval_set=[(X, y)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tm", choices=["approx", "gpu_hist"])
    args = parser.parse_args()
    main(args)
