import xgboost as xgb
import pytest
import testing as tm
import numpy as np


def verify_booster_early_stop(booster):
    dump = booster.get_dump(dump_format='json')
    assert len(dump) == 10      # boosted for 10 rounds.


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = xgb.DMatrix(X, y)
    booster = xgb.train({'objective': 'binary:logistic'}, m,
                        evals=[(m, 'Train')],
                        num_boost_round=1000,
                        early_stopping_rounds=5,
                        verbose_eval=False)
    verify_booster_early_stop(booster)


def eval_error_metric(label, predt):
    r = np.zeros(predt.shape)
    gt = predt > 0.5
    r[gt] = 1 - label[gt]
    le = predt <= 0.5
    r[le] = label[le]
    return np.sum(r)


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping_custom_eval():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = xgb.DMatrix(X, y)
    booster = xgb.train({'objective': 'binary:logistic'}, m,
                        evals=[(m, 'Train')],
                        num_boost_round=1000,
                        early_stopping_rounds=5,
                        verbose_eval=False)
    verify_booster_early_stop(booster)


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping_skl():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    cls = xgb.XGBClassifier()
    cls.fit(X, y, eval_set=[(X, y)], early_stopping_rounds=5)
    booster = cls.get_booster()
    verify_booster_early_stop(booster)


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping_custom_eval_skl():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    cls = xgb.XGBClassifier()
    cls.fit(X, y, eval_set=[(X, y)], early_stopping_rounds=5)
    booster = cls.get_booster()
    verify_booster_early_stop(booster)
