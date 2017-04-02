import numpy as np
import xgboost as xgb
import unittest

dpath = 'demo/data/'
dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

rng = np.random.RandomState(1994)


class TestModels(unittest.TestCase):
    def test_glm(self):
        param = {'silent': 1, 'objective': 'binary:logistic',
                 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1, 'nthread': 1}
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 4
        bst = xgb.train(param, dtrain, num_round, watchlist)
        assert isinstance(bst, xgb.core.Booster)
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.2

    def test_dart(self):
        dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
        dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')
        param = {'max_depth': 5, 'objective': 'binary:logistic', 'booster': 'dart', 'silent': False}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest, ntree_limit=num_round)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        # save dmatrix into binary buffer
        dtest.save_binary('dtest.buffer')
        # save model
        bst.save_model('xgb.model.dart')
        # load model and data in
        bst2 = xgb.Booster(params=param, model_file='xgb.model.dart')
        dtest2 = xgb.DMatrix('dtest.buffer')
        preds2 = bst2.predict(dtest2, ntree_limit=num_round)
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

        # check whether sample_type and normalize_type work
        num_round = 50
        param['silent'] = True
        param['learning_rate'] = 0.1
        param['rate_drop'] = 0.1
        preds_list = []
        for p in [[p0, p1] for p0 in ['uniform', 'weighted'] for p1 in ['tree', 'forest']]:
            param['sample_type'] = p[0]
            param['normalize_type'] = p[1]
            bst = xgb.train(param, dtrain, num_round, watchlist)
            preds = bst.predict(dtest, ntree_limit=num_round)
            err = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1
            preds_list.append(preds)

        for ii in range(len(preds_list)):
            for jj in range(ii + 1, len(preds_list)):
                assert np.sum(np.abs(preds_list[ii] - preds_list[jj])) > 0

    def test_eta_decay(self):
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 4

        # learning_rates as a list
        # init eta with 0 to check whether learning_rates work
        param = {'max_depth': 2, 'eta': 0, 'silent': 1, 'objective': 'binary:logistic'}
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist, learning_rates=[0.8, 0.7, 0.6, 0.5],
                        evals_result=evals_result)
        eval_errors = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should decrease, if eta > 0
        assert eval_errors[0] > eval_errors[-1]

        # init learning_rate with 0 to check whether learning_rates work
        param = {'max_depth': 2, 'learning_rate': 0, 'silent': 1, 'objective': 'binary:logistic'}
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist, learning_rates=[0.8, 0.7, 0.6, 0.5],
                        evals_result=evals_result)
        eval_errors = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should decrease, if learning_rate > 0
        assert eval_errors[0] > eval_errors[-1]

        # check if learning_rates override default value of eta/learning_rate
        param = {'max_depth': 2, 'silent': 1, 'objective': 'binary:logistic'}
        evals_result = {}
        bst = xgb.train(param, dtrain, num_round, watchlist, learning_rates=[0, 0, 0, 0],
                        evals_result=evals_result)
        eval_errors = list(map(float, evals_result['eval']['error']))
        assert isinstance(bst, xgb.core.Booster)
        # validation error should not decrease, if eta/learning_rate = 0
        assert eval_errors[0] == eval_errors[-1]

        # learning_rates as a customized decay function
        def eta_decay(ithround, num_boost_round):
            return num_boost_round / (ithround + 1)

        bst = xgb.train(param, dtrain, num_round, watchlist, learning_rates=eta_decay)
        assert isinstance(bst, xgb.core.Booster)

    def test_custom_objective(self):
        param = {'max_depth': 2, 'eta': 1, 'silent': 1}
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2

        def logregobj(preds, dtrain):
            labels = dtrain.get_label()
            preds = 1.0 / (1.0 + np.exp(-preds))
            grad = preds - labels
            hess = preds * (1.0 - preds)
            return grad, hess

        def evalerror(preds, dtrain):
            labels = dtrain.get_label()
            return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

        # test custom_objective in training
        bst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror)
        assert isinstance(bst, xgb.core.Booster)
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.1

        # test custom_objective in cross-validation
        xgb.cv(param, dtrain, num_round, nfold=5, seed=0,
               obj=logregobj, feval=evalerror)

        # test maximize parameter
        def neg_evalerror(preds, dtrain):
            labels = dtrain.get_label()
            return 'error', float(sum(labels == (preds > 0.0))) / len(labels)

        bst2 = xgb.train(param, dtrain, num_round, watchlist, logregobj, neg_evalerror, maximize=True)
        preds2 = bst2.predict(dtest)
        err2 = sum(1 for i in range(len(preds2))
                   if int(preds2[i] > 0.5) != labels[i]) / float(len(preds2))
        assert err == err2

    def test_multi_eval_metric(self):
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        param = {'max_depth': 2, 'eta': 0.2, 'silent': 1, 'objective': 'binary:logistic'}
        param['eval_metric'] = ["auc", "logloss", 'error']
        evals_result = {}
        bst = xgb.train(param, dtrain, 4, watchlist, evals_result=evals_result)
        assert isinstance(bst, xgb.core.Booster)
        assert len(evals_result['eval']) == 3
        assert set(evals_result['eval'].keys()) == {'auc', 'error', 'logloss'}

    def test_fpreproc(self):
        param = {'max_depth': 2, 'eta': 1, 'silent': 1,
                 'objective': 'binary:logistic'}
        num_round = 2

        def fpreproc(dtrain, dtest, param):
            label = dtrain.get_label()
            ratio = float(np.sum(label == 0)) / np.sum(label == 1)
            param['scale_pos_weight'] = ratio
            return (dtrain, dtest, param)

        xgb.cv(param, dtrain, num_round, nfold=5,
               metrics={'auc'}, seed=0, fpreproc=fpreproc)

    def test_show_stdv(self):
        param = {'max_depth': 2, 'eta': 1, 'silent': 1,
                 'objective': 'binary:logistic'}
        num_round = 2
        xgb.cv(param, dtrain, num_round, nfold=5,
               metrics={'error'}, seed=0, show_stdv=False)

    def test_feature_names_validation(self):
        X = np.random.random((10, 3))
        y = np.random.randint(2, size=(10,))

        dm1 = xgb.DMatrix(X, y)
        dm2 = xgb.DMatrix(X, y, feature_names=("a", "b", "c"))

        bst = xgb.train([], dm1)
        bst.predict(dm1)  # success
        self.assertRaises(ValueError, bst.predict, dm2)
        bst.predict(dm1)  # success

        bst = xgb.train([], dm2)
        bst.predict(dm2)  # success
        self.assertRaises(ValueError, bst.predict, dm1)
        bst.predict(dm2)  # success
