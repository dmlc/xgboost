import numpy as np
import xgboost as xgb
import unittest
import os
import json
import testing as tm
import pytest
import locale
import tempfile

dpath = os.path.join(tm.PROJECT_ROOT, 'demo/data/')
dtrain = xgb.DMatrix(dpath + 'agaricus.txt.train')
dtest = xgb.DMatrix(dpath + 'agaricus.txt.test')

rng = np.random.RandomState(1994)


def json_model(model_path, parameters):
    X = np.random.random((10, 3))
    y = np.random.randint(2, size=(10,))

    dm1 = xgb.DMatrix(X, y)

    bst = xgb.train(parameters, dm1)
    bst.save_model(model_path)

    with open(model_path, 'r') as fd:
        model = json.load(fd)
    return model


class TestModels(unittest.TestCase):
    def test_glm(self):
        param = {'verbosity': 0, 'objective': 'binary:logistic',
                 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1,
                 'nthread': 1}
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
        param = {'max_depth': 5, 'objective': 'binary:logistic',
                 'eval_metric': 'logloss', 'booster': 'dart', 'verbosity': 1}
        # specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest, ntree_limit=num_round)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            dtest_path = os.path.join(tmpdir, 'dtest.dmatrix')
            model_path = os.path.join(tmpdir, 'xgboost.model.dart')
            # save dmatrix into binary buffer
            dtest.save_binary(dtest_path)
            model_path = model_path
            # save model
            bst.save_model(model_path)
            # load model and data in
            bst2 = xgb.Booster(params=param, model_file=model_path)
            dtest2 = xgb.DMatrix(dtest_path)

        preds2 = bst2.predict(dtest2, ntree_limit=num_round)

        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

        def my_logloss(preds, dtrain):
            labels = dtrain.get_label()
            return 'logloss', np.sum(
                np.log(np.where(labels, preds, 1 - preds)))

        # check whether custom evaluation metrics work
        bst = xgb.train(param, dtrain, num_round, watchlist,
                        feval=my_logloss)
        preds3 = bst.predict(dtest, ntree_limit=num_round)
        assert all(preds3 == preds)

        # check whether sample_type and normalize_type work
        num_round = 50
        param['verbosity'] = 0
        param['learning_rate'] = 0.1
        param['rate_drop'] = 0.1
        preds_list = []
        for p in [[p0, p1] for p0 in ['uniform', 'weighted']
                  for p1 in ['tree', 'forest']]:
            param['sample_type'] = p[0]
            param['normalize_type'] = p[1]
            bst = xgb.train(param, dtrain, num_round, watchlist)
            preds = bst.predict(dtest, ntree_limit=num_round)
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1
            preds_list.append(preds)

        for ii in range(len(preds_list)):
            for jj in range(ii + 1, len(preds_list)):
                assert np.sum(np.abs(preds_list[ii] - preds_list[jj])) > 0

    def test_boost_from_prediction(self):
        # Re-construct dtrain here to avoid modification
        margined = xgb.DMatrix(dpath + 'agaricus.txt.train')
        bst = xgb.train({'tree_method': 'hist'}, margined, 1)
        predt_0 = bst.predict(margined, output_margin=True)
        margined.set_base_margin(predt_0)
        bst = xgb.train({'tree_method': 'hist'}, margined, 1)
        predt_1 = bst.predict(margined)

        assert np.any(np.abs(predt_1 - predt_0) > 1e-6)

        bst = xgb.train({'tree_method': 'hist'}, dtrain, 2)
        predt_2 = bst.predict(dtrain)
        assert np.all(np.abs(predt_2 - predt_1) < 1e-6)

    def test_custom_objective(self):
        param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:logistic'}
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 10

        def logregobj(preds, dtrain):
            labels = dtrain.get_label()
            preds = 1.0 / (1.0 + np.exp(-preds))
            grad = preds - labels
            hess = preds * (1.0 - preds)
            return grad, hess

        def evalerror(preds, dtrain):
            labels = dtrain.get_label()
            preds = 1.0 / (1.0 + np.exp(-preds))
            return 'error', float(sum(labels != (preds > 0.5))) / len(labels)

        # test custom_objective in training
        bst = xgb.train(param, dtrain, num_round, watchlist, obj=logregobj,
                        feval=evalerror)
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

        bst2 = xgb.train(param, dtrain, num_round, watchlist, logregobj,
                         neg_evalerror, maximize=True)
        preds2 = bst2.predict(dtest)
        err2 = sum(1 for i in range(len(preds2))
                   if int(preds2[i] > 0.5) != labels[i]) / float(len(preds2))
        assert err == err2

    def test_multi_eval_metric(self):
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        param = {'max_depth': 2, 'eta': 0.2, 'verbosity': 1,
                 'objective': 'binary:logistic'}
        param['eval_metric'] = ["auc", "logloss", 'error']
        evals_result = {}
        bst = xgb.train(param, dtrain, 4, watchlist, evals_result=evals_result)
        assert isinstance(bst, xgb.core.Booster)
        assert len(evals_result['eval']) == 3
        assert set(evals_result['eval'].keys()) == {'auc', 'error', 'logloss'}

    def test_fpreproc(self):
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
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
        param = {'max_depth': 2, 'eta': 1, 'verbosity': 0,
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

    def test_model_binary_io(self):
        model_path = 'test_model_binary_io.bin'
        parameters = {'tree_method': 'hist', 'booster': 'gbtree',
                      'scale_pos_weight': '0.5'}
        X = np.random.random((10, 3))
        y = np.random.random((10,))
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(parameters, dtrain, num_boost_round=2)
        bst.save_model(model_path)
        bst = xgb.Booster(model_file=model_path)
        os.remove(model_path)
        config = json.loads(bst.save_config())
        assert float(config['learner']['objective'][
            'reg_loss_param']['scale_pos_weight']) == 0.5

        buf = bst.save_raw()
        from_raw = xgb.Booster()
        from_raw.load_model(buf)

        buf_from_raw = from_raw.save_raw()
        assert buf == buf_from_raw

    def test_model_json_io(self):
        loc = locale.getpreferredencoding(False)
        model_path = 'test_model_json_io.json'
        parameters = {'tree_method': 'hist', 'booster': 'gbtree'}
        j_model = json_model(model_path, parameters)
        assert isinstance(j_model['learner'], dict)

        bst = xgb.Booster(model_file=model_path)

        bst.save_model(fname=model_path)
        with open(model_path, 'r') as fd:
            j_model = json.load(fd)
        assert isinstance(j_model['learner'], dict)

        os.remove(model_path)
        assert locale.getpreferredencoding(False) == loc

    @pytest.mark.skipif(**tm.no_json_schema())
    def test_json_io_schema(self):
        import jsonschema
        model_path = 'test_json_schema.json'
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        doc = os.path.join(path, 'doc', 'model.schema')
        with open(doc, 'r') as fd:
            schema = json.load(fd)
        parameters = {'tree_method': 'hist', 'booster': 'gbtree'}
        jsonschema.validate(instance=json_model(model_path, parameters),
                            schema=schema)
        os.remove(model_path)

        parameters = {'tree_method': 'hist', 'booster': 'dart'}
        jsonschema.validate(instance=json_model(model_path, parameters),
                            schema=schema)
        os.remove(model_path)

        try:
            xgb.train({'objective': 'foo'}, dtrain, num_boost_round=1)
        except ValueError as e:
            e_str = str(e)
            beg = e_str.find('Objective candidate')
            end = e_str.find('Stack trace')
            e_str = e_str[beg: end]
            e_str = e_str.strip()
            splited = e_str.splitlines()
            objectives = [s.split(': ')[1] for s in splited]
            j_objectives = schema['properties']['learner']['properties'][
                'objective']['oneOf']
            objectives_from_schema = set()
            for j_obj in j_objectives:
                objectives_from_schema.add(
                    j_obj['properties']['name']['const'])
            objectives = set(objectives)
            assert objectives == objectives_from_schema

    @pytest.mark.skipif(**tm.no_json_schema())
    def test_json_dump_schema(self):
        import jsonschema

        def validate_model(parameters):
            X = np.random.random((100, 30))
            y = np.random.randint(0, 4, size=(100,))

            parameters['num_class'] = 4
            m = xgb.DMatrix(X, y)

            booster = xgb.train(parameters, m)
            dump = booster.get_dump(dump_format='json')

            for i in range(len(dump)):
                jsonschema.validate(instance=json.loads(dump[i]),
                                    schema=schema)

        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        doc = os.path.join(path, 'doc', 'dump.schema')
        with open(doc, 'r') as fd:
            schema = json.load(fd)

        parameters = {'tree_method': 'hist', 'booster': 'gbtree',
                      'objective': 'multi:softmax'}
        validate_model(parameters)

        parameters = {'tree_method': 'hist', 'booster': 'dart',
                      'objective': 'multi:softmax'}
        validate_model(parameters)

    def test_slice(self):
        from sklearn.datasets import make_classification
        num_classes = 3
        X, y = make_classification(n_samples=1000, n_informative=5,
                                   n_classes=num_classes)
        dtrain = xgb.DMatrix(data=X, label=y)
        num_parallel_tree = 4
        num_boost_round = 16
        total_trees = num_parallel_tree * num_classes * num_boost_round
        booster = xgb.train({
            'num_parallel_tree': 4, 'subsample': 0.5, 'num_class': 3},
                            num_boost_round=num_boost_round, dtrain=dtrain)
        assert len(booster.get_dump()) == total_trees
        beg = 3
        end = 7
        s = slice(beg, end, 1)
        sliced: xgb.Booster = booster[s]

        sliced_trees = (end - beg) * num_parallel_tree * num_classes
        assert sliced_trees == len(sliced.get_dump())

        self.assertRaises(ValueError, lambda: booster[-1: 0])
        self.assertRaises(ValueError, lambda: booster[1:1])
        self.assertRaises(ValueError, lambda: booster[3:0])
        self.assertRaises(ValueError, lambda: booster[0:4:2])
