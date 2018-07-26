# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import unittest
import itertools
import re
import scipy
import scipy.special

dpath = 'demo/data/'
rng = np.random.RandomState(1994)


class TestSHAP(unittest.TestCase):

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
        scores5 = bst.get_score(importance_type='total_cover')
        scores6 = bst.get_score(importance_type='total_gain')
        assert len(scores1) == len(features)
        assert len(scores2) == len(features)
        assert len(scores3) == len(features)
        assert len(scores4) == len(features)
        assert len(scores5) == len(features)
        assert len(scores6) == len(features)

        # check backwards compatibility of get_fscore
        fscores = bst.get_fscore()
        assert scores1 == fscores

        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

        def fn(max_depth, num_rounds):
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

        # for max_depth, num_rounds in itertools.product(range(0, 3), range(1, 5)):
        #     yield fn, max_depth, num_rounds

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
                    if match is not None:
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
            if tree[i]["value"] is not None:
                return tree[i]["value"]
            else:
                ind = tree[i]["feature_index"]
                if z[ind] == 1:
                    if x[ind] < tree[i]["threshold"]:
                        return exp_value_rec(tree, z, x, tree[i]["yes_ind"])
                    else:
                        return exp_value_rec(tree, z, x, tree[i]["no_ind"])
                else:
                    r_yes = tree[tree[i]["yes_ind"]]["cover"] / tree[i]["cover"]
                    out = exp_value_rec(tree, z, x, tree[i]["yes_ind"])
                    val = out * r_yes

                    r_no = tree[tree[i]["no_ind"]]["cover"] / tree[i]["cover"]
                    out = exp_value_rec(tree, z, x, tree[i]["no_ind"])
                    val += out * r_no
                    return val

        def exp_value(trees, z, x):
            return np.sum([exp_value_rec(tree, z, x) for tree in trees])

        def all_subsets(ss):
            return itertools.chain(*map(lambda x: itertools.combinations(ss, x), range(0, len(ss) + 1)))

        def shap_value(trees, x, i, cond=None, cond_value=None):
            M = len(x)
            z = np.zeros(M)
            other_inds = list(set(range(M)) - set([i]))
            if cond is not None:
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
                total += (v2 - v1) / (scipy.special.binom(M - 1, len(subset)) * M)
                z[i] = 0
                z[list(subset)] = 0
            return total

        def shap_values(trees, x):
            vals = [shap_value(trees, x, i) for i in range(len(x))]
            vals.append(exp_value(trees, np.zeros(len(x)), x))
            return np.array(vals)

        def interaction_values(trees, x):
            M = len(x)
            out = np.zeros((M + 1, M + 1))
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j:
                        out[i, j] = interaction_value(trees, x, i, j) / 2
            svals = shap_values(trees, x)
            main_effects = svals - out.sum(1)
            out[np.diag_indices_from(out)] = main_effects
            return out

        def interaction_value(trees, x, i, j):
            M = len(x)
            z = np.zeros(M)
            other_inds = list(set(range(M)) - set([i, j]))

            total = 0.0
            for subset in all_subsets(other_inds):
                if len(subset) > 0:
                    z[list(subset)] = 1
                v00 = exp_value(trees, z, x)
                z[i] = 1
                v10 = exp_value(trees, z, x)
                z[j] = 1
                v11 = exp_value(trees, z, x)
                z[i] = 0
                v01 = exp_value(trees, z, x)
                z[j] = 0
                total += (v11 - v01 - v10 + v00) / (scipy.special.binom(M - 2, len(subset)) * (M - 1))
                z[list(subset)] = 0
            return total

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
        assert np.linalg.norm(brute_force - fast_method[0, :]) < 1e-4

        brute_force = interaction_values(parse_model(bst), X[0, :])
        fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_interactions=True)
        assert np.linalg.norm(brute_force - fast_method[0, :, :]) < 1e-4

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
        assert np.linalg.norm(brute_force - fast_method[0, :]) < 1e-4

        brute_force = interaction_values(parse_model(bst), X[0, :])
        fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_interactions=True)
        assert np.linalg.norm(brute_force - fast_method[0, :, :]) < 1e-4

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
        assert np.linalg.norm(brute_force - fast_method[0, :]) < 1e-4

        brute_force = interaction_values(parse_model(bst), X[0, :])
        brute_force[-1, -1] += base_score
        fast_method = bst.predict(xgb.DMatrix(X[0:1, :]), pred_interactions=True)
        assert np.linalg.norm(brute_force - fast_method[0, :, :]) < 1e-4
