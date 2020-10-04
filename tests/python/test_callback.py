import xgboost as xgb
import pytest
import os
import testing as tm
import tempfile


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


@pytest.mark.skipif(**tm.no_sklearn())
def test_early_stopping_custom_eval():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = xgb.DMatrix(X, y)
    booster = xgb.train({'objective': 'binary:logistic'}, m,
                        evals=[(m, 'Train')],
                        feval=tm.eval_error_metric,
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


def test_learning_rate_scheduler():
    pass


def test_check_point():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    m = xgb.DMatrix(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                      rounds=1,
                                                      name='model')
        xgb.train({'objective': 'binary:logistic'}, m,
                  num_boost_round=10,
                  verbose_eval=False,
                  callbacks=[check_point])
        for i in range(0, 10):
            assert os.path.exists(
                os.path.join(tmpdir, 'model_' + str(i) + '.json'))

        check_point = xgb.callback.TrainingCheckPoint(directory=tmpdir,
                                                      rounds=1,
                                                      as_pickle=True,
                                                      name='model')
        xgb.train({'objective': 'binary:logistic'}, m,
                  num_boost_round=10,
                  verbose_eval=False,
                  callbacks=[check_point])
        for i in range(0, 10):
            assert os.path.exists(
                os.path.join(tmpdir, 'model_' + str(i) + '.pkl'))
