"""
Using feature weights for column sampling
=========================================

    .. versionadded:: 1.3.0

Feature weights bias which columns are sampled as split candidates. The split
search then chooses among those candidates using gain. Split counts depend on
the data and random draws, so the highest-weight feature need not be used most.
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt

import xgboost


def main(args: argparse.Namespace) -> None:
    rng = np.random.RandomState(1994)

    kRows = 4196
    kCols = 10

    X = rng.randn(kRows, kCols)
    y = rng.randn(kRows)
    # Give f0 zero weight and increase the sampling weight toward f9.
    fw = np.arange(kCols, dtype=np.float32)

    dtrain = xgboost.DMatrix(X, y)
    dtrain.set_info(feature_weights=fw)

    # Sample two of the ten features at each node, using the feature weights.
    bst = xgboost.train(
        {"tree_method": "hist", "colsample_bynode": 0.2},
        dtrain,
        num_boost_round=10,
        evals=[(dtrain, "d")],
    )
    # Count selected splits, which reflect both sampling and the split search.
    feature_map = bst.get_score(importance_type="weight")
    features = [f"f{i}" for i in range(kCols)]
    counts = np.array([feature_map.get(feature, 0) for feature in features])

    assert counts[0] == 0
    # Compare groups to show the sampling bias without requiring a strict ranking.
    assert counts[kCols // 2 :].sum() > counts[: kCols // 2].sum()

    if args.plot:
        fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(9, 4))
        axes[0].barh(features, fw)
        axes[0].set(title="Input feature weights", xlabel="Sampling weight")
        axes[1].barh(features, counts)
        axes[1].set(title="Observed feature usage", xlabel="Number of splits")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        type=int,
        default=1,
        help="Set to 0 to disable plotting feature weights and split counts.",
    )
    args = parser.parse_args()
    main(args)
