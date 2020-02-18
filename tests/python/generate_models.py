import xgboost
import numpy as np
import os

kRounds = 2
kRows = 1000
kCols = 4
kForests = 2
kMaxDepth = 2
kClasses = 3

X = np.random.randn(kRows, kCols)
w = np.random.uniform(size=kRows)

version = xgboost.__version__

np.random.seed(1994)
target_dir = 'models'


def booster_bin(model):
    return os.path.join(target_dir,
                        'xgboost-' + version + '.' + model + '.bin')


def booster_json(model):
    return os.path.join(target_dir,
                        'xgboost-' + version + '.' + model + '.json')


def skl_bin(model):
    return os.path.join(target_dir,
                        'xgboost_scikit-' + version + '.' + model + '.bin')


def skl_json(model):
    return os.path.join(target_dir,
                        'xgboost_scikit-' + version + '.' + model + '.json')


def generate_regression_model():
    print('Regression')
    y = np.random.randn(kRows)

    data = xgboost.DMatrix(X, label=y, weight=w)
    booster = xgboost.train({'tree_method': 'hist',
                             'num_parallel_tree': kForests,
                             'max_depth': kMaxDepth},
                            num_boost_round=kRounds, dtrain=data)
    booster.save_model(booster_bin('reg'))
    booster.save_model(booster_json('reg'))

    reg = xgboost.XGBRegressor(tree_method='hist',
                               num_parallel_tree=kForests,
                               max_depth=kMaxDepth,
                               n_estimators=kRounds)
    reg.fit(X, y, w)
    reg.save_model(skl_bin('reg'))
    reg.save_model(skl_json('reg'))


def generate_logistic_model():
    print('Logistic')
    y = np.random.randint(0, 2, size=kRows)
    assert y.max() == 1 and y.min() == 0

    data = xgboost.DMatrix(X, label=y, weight=w)
    booster = xgboost.train({'tree_method': 'hist',
                             'num_parallel_tree': kForests,
                             'max_depth': kMaxDepth,
                             'objective': 'binary:logistic'},
                            num_boost_round=kRounds, dtrain=data)
    booster.save_model(booster_bin('logit'))
    booster.save_model(booster_json('logit'))

    reg = xgboost.XGBClassifier(tree_method='hist',
                                num_parallel_tree=kForests,
                                max_depth=kMaxDepth,
                                n_estimators=kRounds)
    reg.fit(X, y, w)
    reg.save_model(skl_bin('logit'))
    reg.save_model(skl_json('logit'))


def generate_classification_model():
    print('Classification')
    y = np.random.randint(0, kClasses, size=kRows)
    data = xgboost.DMatrix(X, label=y, weight=w)
    booster = xgboost.train({'num_class': kClasses,
                             'tree_method': 'hist',
                             'num_parallel_tree': kForests,
                             'max_depth': kMaxDepth},
                            num_boost_round=kRounds, dtrain=data)
    booster.save_model(booster_bin('cls'))
    booster.save_model(booster_json('cls'))

    cls = xgboost.XGBClassifier(tree_method='hist',
                                num_parallel_tree=kForests,
                                max_depth=kMaxDepth,
                                n_estimators=kRounds)
    cls.fit(X, y, w)
    cls.save_model(skl_bin('cls'))
    cls.save_model(skl_json('cls'))


def generate_ranking_model():
    print('Learning to Rank')
    y = np.random.randint(5, size=kRows)
    w = np.random.uniform(size=20)
    g = np.repeat(50, 20)

    data = xgboost.DMatrix(X, y, weight=w)
    data.set_group(g)
    booster = xgboost.train({'objective': 'rank:ndcg',
                             'num_parallel_tree': kForests,
                             'tree_method': 'hist',
                             'max_depth': kMaxDepth},
                            num_boost_round=kRounds,
                            dtrain=data)
    booster.save_model(booster_bin('ltr'))
    booster.save_model(booster_json('ltr'))

    ranker = xgboost.sklearn.XGBRanker(n_estimators=kRounds,
                                       tree_method='hist',
                                       objective='rank:ndcg',
                                       max_depth=kMaxDepth,
                                       num_parallel_tree=kForests)
    ranker.fit(X, y, g, sample_weight=w)
    ranker.save_model(skl_bin('ltr'))
    ranker.save_model(skl_json('ltr'))


def write_versions():
    versions = {'numpy': np.__version__,
                'xgboost': version}
    with open(os.path.join(target_dir, 'version'), 'w') as fd:
        fd.write(str(versions))


if __name__ == '__main__':
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    generate_regression_model()
    generate_logistic_model()
    generate_classification_model()
    generate_ranking_model()
    write_versions()
