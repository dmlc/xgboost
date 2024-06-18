# pylint: disable=too-many-locals
"""Tests for learning to rank."""
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm


def run_ranking_qid_df(impl: ModuleType, tree_method: str) -> None:
    """Test ranking with qid packed into X."""
    import scipy.sparse
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

    X, y, q, _ = tm.make_ltr(n_samples=128, n_features=2, n_query_groups=8, max_rel=3)

    # pack qid into x using dataframe
    df = impl.DataFrame(X)
    df["qid"] = q
    ranker = xgb.XGBRanker(n_estimators=3, eval_metric="ndcg", tree_method=tree_method)
    ranker.fit(df, y)
    s = ranker.score(df, y)
    assert s > 0.7

    # works with validation datasets as well
    valid_df = df.copy()
    valid_df.iloc[0, 0] = 3.0
    ranker.fit(df, y, eval_set=[(valid_df, y)])

    # same as passing qid directly
    ranker = xgb.XGBRanker(n_estimators=3, eval_metric="ndcg", tree_method=tree_method)
    ranker.fit(X, y, qid=q)
    s1 = ranker.score(df, y)
    assert np.isclose(s, s1)

    # Works with standard sklearn cv
    if tree_method != "gpu_hist":
        # we need cuML for this.
        kfold = StratifiedGroupKFold(shuffle=False)
        results = cross_val_score(ranker, df, y, cv=kfold, groups=df.qid)
        assert len(results) == 5

    # Works with custom metric
    def neg_mse(*args: Any, **kwargs: Any) -> float:
        return -float(mean_squared_error(*args, **kwargs))

    ranker = xgb.XGBRanker(
        n_estimators=3,
        eval_metric=neg_mse,
        tree_method=tree_method,
        disable_default_eval_metric=True,
    )
    ranker.fit(df, y, eval_set=[(valid_df, y)])
    score = ranker.score(valid_df, y)
    assert np.isclose(score, ranker.evals_result()["validation_0"]["neg_mse"][-1])

    # Works with sparse data
    if tree_method != "gpu_hist":
        # no sparse with cuDF
        X_csr = scipy.sparse.csr_matrix(X)
        df = impl.DataFrame.sparse.from_spmatrix(
            X_csr, columns=[str(i) for i in range(X.shape[1])]
        )
        df["qid"] = q
        ranker = xgb.XGBRanker(
            n_estimators=3, eval_metric="ndcg", tree_method=tree_method
        )
        ranker.fit(df, y)
        s2 = ranker.score(df, y)
        assert np.isclose(s2, s)

    with pytest.raises(ValueError, match="Either `group` or `qid`."):
        ranker.fit(df, y, eval_set=[(X, y)])


def run_ranking_categorical(device: str) -> None:
    """Test LTR with categorical features."""
    from sklearn.model_selection import cross_val_score

    X, y = tm.make_categorical(
        n_samples=512, n_features=10, n_categories=3, onehot=False
    )
    rng = np.random.default_rng(1994)
    qid = rng.choice(3, size=y.shape[0])
    qid = np.sort(qid)
    X["qid"] = qid

    ltr = xgb.XGBRanker(enable_categorical=True, device=device)
    ltr.fit(X, y)
    score = ltr.score(X, y)
    assert score > 0.9

    ltr = xgb.XGBRanker(enable_categorical=True, device=device)

    # test using the score function inside sklearn.
    scores = cross_val_score(ltr, X, y)
    for s in scores:
        assert s > 0.7
