# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import unittest
import itertools
import json
import re
import scipy
import scipy.special

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestBasic(unittest.TestCase):

    def test_basic(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        # save dmatrix into binary buffer
        dtest.save_binary('dtest.buffer')
        # save model
        bst.save_model('xgb.model')
        # load model and data in
        bst2 = xgb.Booster(model_file='xgb.model')
        dtest2 = xgb.DMatrix('dtest.buffer')
        preds2 = bst2.predict(dtest2)
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

    def test_record_results(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        result = {}
        res2 = {}
        xgb.train(param, dtrain, num_round, watchlist,
                  callbacks=[xgb.callback.record_evaluation(result)])
        xgb.train(param, dtrain, num_round, watchlist,
                  evals_result=res2)
        assert result['train']['error'][0] < 0.1
        assert res2 == result

    def test_multiclass(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'num_class': 2}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds)) if preds[i] != labels[i]) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        # save dmatrix into binary buffer
        dtest.save_binary('dtest.buffer')
        # save model
        bst.save_model('xgb.model')
        # load model and data in
        bst2 = xgb.Booster(model_file='xgb.model')
        dtest2 = xgb.DMatrix('dtest.buffer')
        preds2 = bst2.predict(dtest2)
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

    def test_dmatrix_init(self):
        data = np.random.randn(5, 5)

        # different length
        self.assertRaises(ValueError, xgb.DMatrix, data,
                          feature_names=list('abcdef'))
        # contains duplicates
        self.assertRaises(ValueError, xgb.DMatrix, data,
                          feature_names=['a', 'b', 'c', 'd', 'd'])
        # contains symbol
        self.assertRaises(ValueError, xgb.DMatrix, data,
                          feature_names=['a', 'b', 'c', 'd', 'e<1'])

        dm = xgb.DMatrix(data)
        dm.feature_names = list('abcde')
        assert dm.feature_names == list('abcde')

        dm.feature_types = 'q'
        assert dm.feature_types == list('qqqqq')

        dm.feature_types = list('qiqiq')
        assert dm.feature_types == list('qiqiq')

        def incorrect_type_set():
            dm.feature_types = list('abcde')

        self.assertRaises(ValueError, incorrect_type_set)

        # reset
        dm.feature_names = None
        self.assertEqual(dm.feature_names, ['f0', 'f1', 'f2', 'f3', 'f4'])
        assert dm.feature_types is None

    def test_feature_names(self):
        data = np.random.randn(100, 5)
        target = np.array([0, 1] * 50)

        cases = [['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
                 [u'要因1', u'要因2', u'要因3', u'要因4', u'要因5']]

        for features in cases:
            dm = xgb.DMatrix(data, label=target,
                             feature_names=features)
            assert dm.feature_names == features
            assert dm.num_row() == 100
            assert dm.num_col() == 5

            params = {'objective': 'multi:softprob',
                      'eval_metric': 'mlogloss',
                      'eta': 0.3,
                      'num_class': 3}

            bst = xgb.train(params, dm, num_boost_round=10)
            scores = bst.get_fscore()
            assert list(sorted(k for k in scores)) == features

            dummy = np.random.randn(5, 5)
            dm = xgb.DMatrix(dummy, feature_names=features)
            bst.predict(dm)

            # different feature name must raises error
            dm = xgb.DMatrix(dummy, feature_names=list('abcde'))
            self.assertRaises(ValueError, bst.predict, dm)

    def test_feature_importances(self):
        data = np.random.randn(100, 5)
        target = np.array([0, 1] * 50)

        features = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']

        dm = xgb.DMatrix(data, label=target,
                         feature_names=features)
        params = {'objective': 'multi:softprob',
                  'eval_metric': 'mlogloss',
                  'eta': 0.3,
                  'num_class': 3}

        bst = xgb.train(params, dm, num_boost_round=10)

        # number of feature importances should == number of features
        scores1 = bst.get_score()
        scores2 = bst.get_score(importance_type='weight')
        scores3 = bst.get_score(importance_type='cover')
        scores4 = bst.get_score(importance_type='gain')
        assert len(scores1) == len(features)
        assert len(scores2) == len(features)
        assert len(scores3) == len(features)
        assert len(scores4) == len(features)

        # check backwards compatibility of get_fscore
        fscores = bst.get_fscore()
        assert scores1 == fscores

    def test_dump(self):
        data = np.random.randn(100, 2)
        target = np.array([0, 1] * 50)
        features = ['Feature1', 'Feature2']

        dm = xgb.DMatrix(data, label=target, feature_names=features)
        params = {'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'eta': 0.3,
                  'max_depth': 1}

        bst = xgb.train(params, dm, num_boost_round=1)

        # number of feature importances should == number of features
        dump1 = bst.get_dump()
        self.assertEqual(len(dump1), 1, "Expected only 1 tree to be dumped.")
        self.assertEqual(len(dump1[0].splitlines()), 3,
                         "Expected 1 root and 2 leaves - 3 lines in dump.")

        dump2 = bst.get_dump(with_stats=True)
        self.assertEqual(dump2[0].count('\n'), 3,
                         "Expected 1 root and 2 leaves - 3 lines in dump.")
        self.assertGreater(dump2[0].find('\n'), dump1[0].find('\n'),
                           "Expected more info when with_stats=True is given.")

        dump3 = bst.get_dump(dump_format="json")
        dump3j = json.loads(dump3[0])
        self.assertEqual(dump3j["nodeid"], 0, "Expected the root node on top.")

        dump4 = bst.get_dump(dump_format="json", with_stats=True)
        dump4j = json.loads(dump4[0])
        self.assertIn("gain", dump4j, "Expected 'gain' to be dumped in JSON.")

    def test_load_file_invalid(self):
        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
                          model_file='incorrect_path')

        self.assertRaises(xgb.core.XGBoostError, xgb.Booster,
                          model_file=u'不正なパス')

    def test_dmatrix_numpy_init_omp(self):

        rows = [1000, 11326, 15000]
        cols = 50
        for row in rows:
            X = np.random.randn(row, cols)
            y = np.random.randn(row).astype('f')
            dm = xgb.DMatrix(X, y, nthread=0)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

            dm = xgb.DMatrix(X, y, nthread=10)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

    def test_dmatrix_numpy_init(self):
        data = np.random.randn(5, 5)
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 5
        assert dm.num_col() == 5

        data = np.matrix([[1, 2], [3, 4]])
        dm = xgb.DMatrix(data)
        assert dm.num_row() == 2
        assert dm.num_col() == 2

        # 0d array
        self.assertRaises(ValueError, xgb.DMatrix, np.array(1))
        # 1d array
        self.assertRaises(ValueError, xgb.DMatrix, np.array([1, 2, 3]))
        # 3d array
        data = np.random.randn(5, 5, 5)
        self.assertRaises(ValueError, xgb.DMatrix, data)
        # object dtype
        data = np.array([['a', 'b'], ['c', 'd']])
        self.assertRaises(ValueError, xgb.DMatrix, data)

    def test_cv(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_no_shuffle(self):
        dm = xgb.DMatrix(dpath + 'agaricus.txt.train')
        params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, shuffle=False, nfold=10, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)


def test_contributions():
    dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
    dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

    def test_fn(max_depth, num_rounds):
        # train
        params = {'max_depth': max_depth, 'eta': 1, 'silent': 1}
        bst = xgb.train(params, dtrain, num_boost_round=num_rounds)

        # predict
        preds = bst.predict(dtest)
        contribs = bst.predict(dtest, pred_contribs=True)

        # result should be (number of features + BIAS) * number of rows
        assert contribs.shape == (dtest.num_row(), dtest.num_col() + 1)

        # sum of contributions should be same as predictions
        np.testing.assert_array_almost_equal(np.sum(contribs, axis=1), preds)

    for max_depth, num_rounds in itertools.product(range(0, 3), range(1, 5)):
        yield test_fn, max_depth, num_rounds

    # check that we get the right SHAP values for a basic AND example
    # (https://arxiv.org/abs/1706.06060)
    X = np.zeros((4, 2))
    X[0, :] = 1
    X[1, 0] = 1
    X[2, 1] = 1
    y = np.zeros(4)
    y[0] = 1
    param = {"max_depth": 2, "base_score": 0.0, "eta": 1.0, "lambda": 0}
    bst = xgb.train(param, xgb.DMatrix(X, label=y), 1)
    out = bst.predict(xgb.DMatrix(X[0:1, :]), pred_contribs=True)
    assert out[0, 0] == 0.375
    assert out[0, 1] == 0.375
    assert out[0, 2] == 0.25


    def parse_model(model):
        trees = []
        r_exp = r"([0-9]+):\[f([0-9]+)<([0-9\.e-]+)\] yes=([0-9]+),no=([0-9]+).*cover=([0-9e\.]+)"
        r_exp_leaf = r"([0-9]+):leaf=([0-9\.e-]+),cover=([0-9e\.]+)"
        for tree in model.get_dump(with_stats=True):

            lines = list(tree.splitlines())
            trees.append([None for i in range(len(lines))])
            for line in lines:
                match = re.search(r_exp, line)
                if match != None:
                    ind = int(match.group(1))
                    while ind >= len(trees[-1]):
                        trees[-1].append(None)
                    trees[-1][ind] = {
                        "yes_ind": int(match.group(4)),
                        "no_ind": int(match.group(5)),
                        "value": None,
                        "threshold": float(match.group(3)),
                        "feature_index": int(match.group(2)),
                        "cover": float(match.group(6))
                    }
                else:

                    match = re.search(r_exp_leaf, line)
                    ind = int(match.group(1))
                    while ind >= len(trees[-1]):
                        trees[-1].append(None)
                    trees[-1][ind] = {
                        "value": float(match.group(2)),
                        "cover": float(match.group(3))
                    }
        return trees

    def exp_value_rec(tree, z, x, i=0):
        if tree[i]["value"] != None:
            return tree[i]["value"]
        else:
            ind = tree[i]["feature_index"]
            if z[ind] == 1:
                if x[ind] < tree[i]["threshold"]:
                    return exp_value_rec(tree, z, x, tree[i]["yes_ind"])
                else:
                    return exp_value_rec(tree, z, x, tree[i]["no_ind"])
            else:
                r_yes = tree[tree[i]["yes_ind"]]["cover"]/tree[i]["cover"]
                out = exp_value_rec(tree, z, x, tree[i]["yes_ind"])
                val = out*r_yes

                r_no = tree[tree[i]["no_ind"]]["cover"]/tree[i]["cover"]
                out = exp_value_rec(tree, z, x, tree[i]["no_ind"])
                val += out*r_no
                return val

    def exp_value(trees, z, x):
        return np.sum([exp_value_rec(tree, z, x) for tree in trees])

    def all_subsets(ss):
        return itertools.chain(*map(lambda x: itertools.combinations(ss, x), range(0, len(ss)+1)))

    def shap_value(trees, x, i, cond=None, cond_value=None):
        M = len(x)
        z = np.zeros(M)
        other_inds = list(set(range(M)) - set([i]))
        if cond != None:
            other_inds = list(set(other_inds) - set([cond]))
            z[cond] = cond_value
            M -= 1
        total = 0.0

        for subset in all_subsets(other_inds):
            if len(subset) > 0:
                z[list(subset)] = 1
            v1 = exp_value(trees, z, x)
            z[i] = 1
            v2 = exp_value(trees, z, x)
            total += (v2 - v1)/(scipy.special.binom(M-1,len(subset))*M)
            z[i] = 0
            z[list(subset)] = 0
        return total

    def shap_values(trees, x):
        vals = [shap_value(trees, x, i) for i in range(len(x))]
        vals.append(exp_value(trees, np.zeros(len(x)), x))
        return np.array(vals)

    def interaction_value(trees, x, i, j):
        with_i = shap_value(parse_model(bst), X[0, :], i, j, 1)
        without_i = shap_value(parse_model(bst), X[0, :], i, j, 0)
        return with_i - without_i

    def interaction_values(trees, x):
        M = len(x)
        out = np.zeros((M, M))
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    out[i,j] = interaction_value(trees, x, i, j)
        return out

    # test a simple and function
    M = 2
    N = 4
    X = np.zeros((N, M))
    X[0, :] = 1
    X[1, 0] = 1
    X[2, 1] = 1
    y = np.zeros(N)
    y[0] = 1
    param = {"max_depth": 2, "base_score": 0.0, "eta": 1.0, "lambda": 0}
    bst = xgb.train(param, xgb.DMatrix(X, label=y), 1)
    brute_force = shap_values(parse_model(bst), X[0, :])
    fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_contribs=True)
    assert np.linalg.norm(brute_force-fast_method[0,:]) < 1e-4

    brute_force = interaction_values(parse_model(bst), X[0, :])
    fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_contribs=True, interaction_contribs=True)
    assert np.linalg.norm(brute_force - fast_method[0,:,:]) < 1e-4

    # test a random function
    np.random.seed(0)
    M = 2
    N = 4
    X = np.random.randn(N, M)
    y = np.random.randn(N)
    param = {"max_depth": 2, "base_score": 0.0, "eta": 1.0, "lambda": 0}
    bst = xgb.train(param, xgb.DMatrix(X, label=y), 1)
    brute_force = shap_values(parse_model(bst), X[0, :])
    fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_contribs=True)
    assert np.linalg.norm(brute_force-fast_method[0,:]) < 1e-4

    brute_force = interaction_values(parse_model(bst), X[0, :])
    fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_contribs=True, interaction_contribs=True)
    assert np.linalg.norm(brute_force - fast_method[0,:,:]) < 1e-4

    # test another larger more complex random function
    np.random.seed(0)
    M = 5
    N = 100
    X = np.random.randn(N, M)
    y = np.random.randn(N)
    base_score = 1.0
    param = {"max_depth": 5, "base_score": base_score, "eta": 0.1, "gamma": 2.0}
    bst = xgb.train(param, xgb.DMatrix(X, label=y), 10)
    brute_force = shap_values(parse_model(bst), X[0, :])
    brute_force[-1] += base_score
    fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_contribs=True)
    assert np.linalg.norm(brute_force-fast_method[0,:]) < 1e-4

    brute_force = interaction_values(parse_model(bst), X[0, :])
    fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_contribs=True, interaction_contribs=True)
    assert np.linalg.norm(brute_force - fast_method[0,:,:]) < 1e-4
