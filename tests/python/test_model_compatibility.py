import xgboost
import os
import generate_models as gm


def test_model_compability():
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'models')
    models = [
        os.path.join(root, f) for root, subdir, files in os.walk(path)
        for f in files
        if f != 'version'
    ]
    assert len(models) == 12

    for path in models:
        name = os.path.basename(path)
        if name.startswith('xgboost-'):
            booster = xgboost.Booster(model_file=path)
            if name.find('cls') != -1:
                assert (len(booster.get_dump()) ==
                        gm.kForests * gm.kRounds * gm.kClasses)
            else:
                assert len(booster.get_dump()) == gm.kForests * gm.kRounds
        elif name.startswith('xgboost_scikit'):
            if name.find('reg') != -1:
                reg = xgboost.XGBRegressor()
                reg.load_model(path)
                assert (len(reg.get_booster().get_dump()) ==
                        gm.kRounds * gm.kForests)
            elif name.find('cls') != -1:
                cls = xgboost.XGBClassifier()
                cls.load_model(path)
                assert len(cls.classes_) == gm.kClasses
                assert len(cls._le.classes_) == gm.kClasses
                assert cls.n_classes_ == gm.kClasses
                assert (len(cls.get_booster().get_dump()) ==
                        gm.kRounds * gm.kForests * gm.kClasses), path
            elif name.find('ltr') != -1:
                ltr = xgboost.XGBRanker()
                ltr.load_model(path)
                assert (len(ltr.get_booster().get_dump()) ==
                        gm.kRounds * gm.kForests)
            else:
                assert False
        else:
            assert False
