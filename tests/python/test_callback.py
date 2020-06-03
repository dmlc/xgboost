import xgboost as xgb
import pytest
import testing as tm
import numpy as np


def verify_booster_early_stop(booster, early_stop):
    dump = booster.get_dump(dump_format='json')
    assert early_stop.current_rounds == 5  # no improvement in 5 rounds
    assert len(dump) == 10      # boosted for 10 rounds.

    assert (len(early_stop.history['Train']['eval-error']) -
            len(early_stop.best_scores['Train']['eval-error']) == 5)
    eval_error = early_stop.history['Train']['eval-error']
    eval_error = eval_error[5:]
    for e in eval_error:
        assert e == eval_error[0]


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = xgb.DMatrix(X, y)
    early_stop = xgb.callback.EarlyStopping(data=m, name='Train', rounds=5)
    booster = xgb.train({'objective': 'binary:logistic'}, m,
                        num_boost_round=1000,
                        verbose_eval=False,
                        callbacks=[early_stop])
    verify_booster_early_stop(booster, early_stop)


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
    early_stop = xgb.callback.EarlyStopping(data=X, label=y,
                                            name='Train',
                                            rounds=5,
                                            metric=eval_error_metric,
                                            metric_name='eval-error')
    booster = xgb.train({'objective': 'binary:logistic'}, m,
                        num_boost_round=1000,
                        verbose_eval=False,
                        callbacks=[early_stop])
    verify_booster_early_stop(booster, early_stop)


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping_skl():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    early_stop = xgb.callback.EarlyStopping(data=X, name='Train', rounds=5,
                                            label=y)
    cls = xgb.XGBClassifier()
    cls.fit(X, y, callbacks=[early_stop])
    booster = cls.get_booster()
    verify_booster_early_stop(booster, early_stop)


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping_custom_eval_skl():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    early_stop = xgb.callback.EarlyStopping(data=X, label=y,
                                            name='Train',
                                            rounds=5,
                                            metric=eval_error_metric,
                                            metric_name='eval-error')
    cls = xgb.XGBClassifier()
    cls.fit(X, y, callbacks=[early_stop])
    booster = cls.get_booster()
    verify_booster_early_stop(booster, early_stop)
