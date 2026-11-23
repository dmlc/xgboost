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
