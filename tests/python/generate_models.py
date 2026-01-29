import os

import numpy as np
from sklearn.datasets import make_classification

import xgboost
from xgboost.testing import make_categorical, make_ltr

kRounds = 4
kRows = 1000
kCols = 4
kForests = 2
kMaxDepth = 3
kClasses = 3


version = xgboost.__version__

target_dir = "models"


def booster_ubj(model: str) -> str:
    return os.path.join(target_dir, "xgboost-" + version + "." + model + ".ubj")


def booster_json(model: str) -> str:
    return os.path.join(target_dir, "xgboost-" + version + "." + model + ".json")


def skl_ubj(model: str) -> str:
    return os.path.join(target_dir, "xgboost_scikit-" + version + "." + model + ".ubj")


def skl_json(model: str) -> str:
    return os.path.join(target_dir, "xgboost_scikit-" + version + "." + model + ".json")


def generate_regression_model() -> None:
    print("Regression")
    X, y = make_categorical(
        n_samples=kRows, n_features=kCols, n_categories=16, onehot=False, cat_ratio=0.5
    )
    w = np.random.default_rng(2025).uniform(size=X.shape[0])
    data = xgboost.DMatrix(X, label=y, weight=w, enable_categorical=True)
    booster = xgboost.train(
        {
            "tree_method": "hist",
            "num_parallel_tree": kForests,
            "max_depth": kMaxDepth,
            "base_score": 0.5,
        },
        num_boost_round=kRounds,
        dtrain=data,
    )
    booster.save_model(booster_ubj("reg"))
    booster.save_model(booster_json("reg"))

    reg = xgboost.XGBRegressor(
        tree_method="hist",
        num_parallel_tree=kForests,
        max_depth=kMaxDepth,
        n_estimators=kRounds,
        base_score=0.5,
        enable_categorical=True,
    )
    reg.fit(X, y, sample_weight=w)
    reg.save_model(skl_ubj("reg"))
    reg.save_model(skl_json("reg"))


def generate_logistic_model() -> None:
    print("Logistic")
    X, y = make_classification(n_samples=kRows, n_features=kCols, random_state=2025)
    assert y.max() == 1 and y.min() == 0
    w = np.random.default_rng(2025).uniform(size=X.shape[0])

    for objective, name in [
        ("binary:logistic", "logit"),
        ("binary:logitraw", "logitraw"),
    ]:
        data = xgboost.DMatrix(X, label=y, weight=w)
        booster = xgboost.train(
            {
                "tree_method": "hist",
                "num_parallel_tree": kForests,
                "max_depth": kMaxDepth,
                "objective": objective,
                "base_score": 0.5,
            },
            num_boost_round=kRounds,
            dtrain=data,
        )
        booster.save_model(booster_ubj(name))
        booster.save_model(booster_json(name))

        reg = xgboost.XGBClassifier(
            tree_method="hist",
            num_parallel_tree=kForests,
            max_depth=kMaxDepth,
            n_estimators=kRounds,
            objective=objective,
            base_score=0.5,
        )
        reg.fit(X, y, sample_weight=w)
        reg.save_model(skl_ubj(name))
        reg.save_model(skl_json(name))


def generate_classification_model() -> None:
    print("Classification")
    X, y = make_classification(
        n_samples=kRows,
        n_features=kCols,
        random_state=2025,
        n_classes=kClasses,
        n_informative=4,
        n_redundant=0,
    )
    w = np.random.default_rng(2025).uniform(size=X.shape[0])

    data = xgboost.DMatrix(X, label=y, weight=w)
    booster = xgboost.train(
        {
            "num_class": kClasses,
            "tree_method": "hist",
            "num_parallel_tree": kForests,
            "max_depth": kMaxDepth,
        },
        num_boost_round=kRounds,
        dtrain=data,
    )
    booster.save_model(booster_ubj("cls"))
    booster.save_model(booster_json("cls"))

    cls = xgboost.XGBClassifier(
        tree_method="hist",
        num_parallel_tree=kForests,
        max_depth=kMaxDepth,
        n_estimators=kRounds,
    )
    cls.fit(X, y, sample_weight=w)
    cls.save_model(skl_ubj("cls"))
    cls.save_model(skl_json("cls"))


def generate_ranking_model() -> None:
    print("Learning to Rank")
    X, y, qid, w = make_ltr(
        n_samples=kRows, n_features=kCols, n_query_groups=7, max_rel=3
    )

    data = xgboost.DMatrix(X, y, weight=w, qid=qid)
    booster = xgboost.train(
        {
            "objective": "rank:ndcg",
            "num_parallel_tree": kForests,
            "tree_method": "hist",
            "max_depth": kMaxDepth,
            "base_score": 0.5,
        },
        num_boost_round=kRounds,
        dtrain=data,
    )
    booster.save_model(booster_ubj("ltr"))
    booster.save_model(booster_json("ltr"))

    ranker = xgboost.sklearn.XGBRanker(
        n_estimators=kRounds,
        tree_method="hist",
        objective="rank:ndcg",
        max_depth=kMaxDepth,
        num_parallel_tree=kForests,
        base_score=0.5,
    )
    ranker.fit(X, y, qid=qid, sample_weight=w)
    ranker.save_model(skl_ubj("ltr"))
    ranker.save_model(skl_json("ltr"))


def generate_aft_survival_models() -> None:
    print("AFT Survival")
    X, y_lower = make_categorical(
        n_samples=kRows, n_features=kCols, n_categories=16, onehot=False, cat_ratio=0.5
    )
    w = np.random.default_rng(2025).uniform(size=X.shape[0])
    y_upper = y_lower + np.mean(y_lower) + w
    data = xgboost.QuantileDMatrix(
        X, label_lower_bound=y_lower, label_upper_bound=y_upper, enable_categorical=True
    )
    params = {
        "num_parallel_tree": kForests,
        "tree_method": "hist",
        "max_depth": kMaxDepth,
        "objective": "survival:aft",
        "aft_loss_distribution": "normal",
        "base_score": 0.5,
    }
    booster = xgboost.train(params, num_boost_round=kRounds, dtrain=data)
    booster.save_model(booster_ubj("aft"))
    booster.save_model(booster_json("aft"))


if __name__ == "__main__":
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    generate_regression_model()
    generate_logistic_model()
    generate_classification_model()
    generate_ranking_model()
    generate_aft_survival_models()
