import sys

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

import xgboost as xgb


def run_omp(output_path: str):
    X, y = make_classification(
        n_samples=200, n_features=32, n_classes=3, n_informative=8
    )
    Xy = xgb.DMatrix(X, y, nthread=16)
    booster = xgb.train(
        {"num_class": 3, "objective": "multi:softprob", "n_jobs": 16},
        Xy,
        num_boost_round=8,
    )
    score = booster.predict(Xy)
    auc = roc_auc_score(y, score, average="weighted", multi_class="ovr")
    with open(output_path, "w") as fd:
        fd.write(str(auc))


if __name__ == "__main__":
    out = sys.argv[1]
    run_omp(out)
