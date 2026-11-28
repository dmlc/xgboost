import numpy as np
import pytest

import xgboost as xgb


def evaluate(metric: str, labels, predictions, groups=None, weights=None) -> float:
    data = xgb.DMatrix(np.zeros((len(labels), 1)), label=labels)
    if groups is not None:
        data.set_group(groups)
    if weights is not None:
        data.set_weight(weights)
    model = xgb.train(
        {"eval_metric": metric, "nthread": 2, "base_score": 0.5},
        data,
        num_boost_round=0,
    )
    data.set_base_margin(np.asarray(predictions, dtype=np.float32))
    return float(model.eval(data).rsplit(":", 1)[1])


def test_recall_values() -> None:
    assert evaluate("recall@2", [1] * 5, [5, 4, 3, 2, 1]) == pytest.approx(0.4)
    assert evaluate("recall@2", [0, 0, 1, 1], [1] * 4) == 0
    assert evaluate("recall@2", [1, 1, 0, 0], [1] * 4) == 1
    assert evaluate("recall", [0] * 40 + [1] * 40, list(range(80))) == 1
    assert evaluate(
        "recall@1", [1, 1, 0, 1], [3, 2, 2, 1], [2, 2], [2, 1]
    ) == pytest.approx(1 / 3)
    assert evaluate("recall@2", [0, 0], [2, 1]) == 1
    assert evaluate("recall@2-", [0, 0], [2, 1]) == 0
    assert evaluate("recall@1", [0, 0], [2, 1], [2], [0]) == 0
    assert evaluate("recall@2", [0.0000005, 0.9999995, 1], [3, 2, 1]) == pytest.approx(
        0.5
    )


@pytest.mark.parametrize("label", [0.5, -1.0, 2.0, float("nan"), float("inf")])
def test_recall_invalid_labels(label: float) -> None:
    with pytest.raises(xgb.core.XGBoostError):
        evaluate("recall@1", [label, 0], [2, 1])


@pytest.mark.parametrize("name", ["recall", "recall-", "recall@2", "recall@2-"])
@pytest.mark.parametrize("maximize", [None, True, False])
def test_recall_early_stopping_direction(name: str, maximize) -> None:
    model = xgb.Booster()
    callback = xgb.callback.EarlyStopping(rounds=2, maximize=maximize)
    history = {"valid": {name: []}}
    scores = [0.2, 0.5, 0.3, 0.4]
    for iteration, score in enumerate(scores):
        history["valid"][name].append(score)
        if callback.after_iteration(model, iteration, history):
            break
    expected = 0 if maximize is False else 1
    assert int(model.attr("best_iteration")) == expected
    assert callback.maximize is (maximize is not False)


def test_recall_train_cv_and_ranker() -> None:
    rng = np.random.default_rng(10)
    features = rng.normal(size=(24, 3))
    labels = np.tile([1, 0, 1, 0], 6)
    data = xgb.DMatrix(features, label=labels)
    data.set_group([4] * 6)
    params = {"objective": "rank:pairwise", "eval_metric": "recall@2", "nthread": 2}
    history = {}
    model = xgb.train(
        params,
        data,
        num_boost_round=6,
        evals=[(data, "train")],
        evals_result=history,
        early_stopping_rounds=2,
        verbose_eval=False,
    )
    values = history["train"]["recall@2"]
    assert model.best_score == max(values)
    result = xgb.cv(
        params,
        data,
        num_boost_round=4,
        nfold=2,
        early_stopping_rounds=2,
        as_pandas=False,
        verbose_eval=False,
    )
    assert "test-recall@2-mean" in result
    pytest.importorskip("sklearn")
    ranker = xgb.XGBRanker(
        objective="rank:pairwise",
        eval_metric="recall@2",
        n_estimators=6,
        early_stopping_rounds=2,
        n_jobs=2,
    )
    ranker.fit(
        features,
        labels,
        group=[4] * 6,
        eval_set=[(features, labels)],
        eval_group=[[4] * 6],
        verbose=False,
    )
    assert ranker.best_score == max(ranker.evals_result()["validation_0"]["recall@2"])


def matrix(labels=(0, 0, 1, 0), groups=(4,), weights=None):
    data = xgb.DMatrix(np.zeros((4, 1)), label=labels)
    if groups is not None:
        data.set_group(groups)
    if weights is not None:
        data.set_weight(weights)
    data.set_base_margin(np.array([4, 3, 2, 1], dtype=np.float32))
    return data


def model(data, metric):
    return xgb.train(
        {"eval_metric": metric, "nthread": 2, "base_score": 0.5},
        data,
        num_boost_round=0,
    )


def score(booster, data):
    return float(booster.eval(data).rsplit(":", 1)[1])


@pytest.mark.parametrize("metric,regrouped", [("recall@1", 1.0), ("recall@1-", 0.5)])
def test_boundary_mutation(metric, regrouped):
    data = matrix()
    booster = model(data, metric)
    assert score(booster, data) == 0.0
    data.set_group([2, 2])
    assert score(booster, data) == regrouped
    data.set_group([4])
    assert score(booster, data) == 0.0


@pytest.mark.parametrize("metric", ["recall", "recall-", "recall@1", "recall@1-"])
def test_invalid_relabel_recovery(metric):
    data = matrix()
    booster = model(data, metric)
    original = score(booster, data)
    data.set_label([2, 0, 1, 0])
    with pytest.raises(
        xgb.core.XGBoostError, match="recall can only be used with binary labels"
    ):
        score(booster, data)
    data.set_label([0, 0, 1, 0])
    assert score(booster, data) == original


def test_weighted_invalid_group_total():
    data = matrix(groups=[2, 2], weights=[1, 2])
    booster = model(data, "recall@1")
    assert score(booster, data) == 1.0
    data.set_group([1, 2])
    with pytest.raises(xgb.core.XGBoostError, match="Invalid query group structure"):
        score(booster, data)
    data.set_group([2, 2])
    assert score(booster, data) == 1.0


def test_valid_relabel_repeat_guard():
    data = matrix(groups=[2, 2])
    booster = model(data, "recall@1")
    for _ in range(3):
        assert score(booster, data) == 1.0
    data.set_label([1, 0, 0, 1])
    assert score(booster, data) == 0.5
    fresh = matrix(labels=[1, 0, 0, 1], groups=[2, 2])
    assert score(model(fresh, "recall@1"), fresh) == 0.5


@pytest.mark.parametrize("weights", [[1], [1, 2, 3]])
def test_bad_weight_count_guard(weights):
    data = matrix(groups=[2, 2])
    booster = model(data, "recall@1")
    score(booster, data)
    data.set_weight(weights)
    with pytest.raises(
        xgb.core.XGBoostError,
        match="Size of weight must equal to the number of query groups",
    ):
        score(booster, data)
    data.set_weight([1, 2])
    assert score(booster, data) == 1.0


def test_implicit_row_weights_guard():
    data = matrix(groups=None, weights=[1, 2, 3, 4])
    booster = model(data, "recall@1")
    with pytest.raises(
        xgb.core.XGBoostError,
        match="Size of weight must equal to the number of query groups",
    ):
        score(booster, data)


@pytest.mark.parametrize(
    "metric,expected", [("recall@1", [0.0, 1.0, 1.0]), ("recall@1-", [0.0, 0.5, 0.5])]
)
def test_group_count_mutation_weighted(metric, expected):
    data = matrix(groups=[2, 2], weights=[1, 2])
    booster = model(data, metric)
    score(booster, data)
    for groups, weights, oracle in zip(
        [[4], [1, 1, 2], [2, 2]], [[3], [1, 2, 3], [2, 2]], expected
    ):
        data.set_group(groups)
        data.set_weight(weights)
        assert score(booster, data) == pytest.approx(oracle)
        fresh = matrix(groups=groups, weights=weights)
        assert score(model(fresh, metric), fresh) == pytest.approx(oracle)


def test_parameter_config_guard():
    data = matrix(groups=[4], labels=[1, 1, 0, 0])
    booster = model(data, "recall@1")
    assert score(booster, data) == 0.5
    booster.set_param({"eval_metric": "recall@2"})
    values = {
        item.rsplit(":", 1)[0].split("-", 1)[1]: float(item.rsplit(":", 1)[1])
        for item in booster.eval(data).split("\t")[1:]
    }
    assert values["recall@1"] == 0.5
    assert values["recall@2"] == 1.0
    config = booster.save_config()
    booster.load_config(config)
    assert booster.save_config() == config
    assert booster.eval(data).endswith("eval-recall@2:1.00000000000000000")


def test_trained_prediction_mutation():
    rng = np.random.default_rng(81)
    train = xgb.DMatrix(rng.normal(size=(20, 2)), label=np.tile([1, 0], 10))
    train.set_group([4] * 5)
    booster = xgb.train(
        {
            "objective": "rank:pairwise",
            "eval_metric": "recall@1",
            "base_score": 0.5,
            "nthread": 2,
            "max_depth": 2,
        },
        train,
        num_boost_round=3,
    )
    data = xgb.DMatrix(
        np.array([[2, 1], [-1, 2], [0, 0], [1, -2]], dtype=np.float32),
        label=[1, 0, 1, 0],
    )
    data.set_group([2, 2])
    score(booster, data)
    data.set_group([1, 3])
    fresh = xgb.DMatrix(
        np.array([[2, 1], [-1, 2], [0, 0], [1, -2]], dtype=np.float32),
        label=[1, 0, 1, 0],
    )
    fresh.set_group([1, 3])
    predictions = booster.predict(data)
    assert np.array_equal(predictions, booster.predict(fresh))
    labels = np.array([1, 0, 1, 0])
    oracle = []
    for begin, end in [(0, 1), (1, 4)]:
        order = np.argsort(-predictions[begin:end], kind="stable")
        relevant = labels[begin:end]
        oracle.append(relevant[order[0]] / relevant.sum() if relevant.sum() else 1.0)
    assert score(booster, data) == pytest.approx(np.mean(oracle))
    assert score(booster, fresh) == pytest.approx(np.mean(oracle))
