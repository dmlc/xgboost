import json
import os
import tempfile
from typing import Optional

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.core import Integer
from xgboost.testing.updater import ResetStrategy

dpath = tm.data_dir(__file__)

rng = np.random.RandomState(1994)


class TestModels:
    def test_glm(self):
        param = {
            "objective": "binary:logistic",
            "booster": "gblinear",
            "alpha": 0.0001,
            "lambda": 1,
            "nthread": 1,
        }
        dtrain, dtest = tm.load_agaricus(__file__)
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        num_round = 4
        bst = xgb.train(param, dtrain, num_round, watchlist)
        assert isinstance(bst, xgb.core.Booster)
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(
            1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
        ) / float(len(preds))
        assert err < 0.2

    def test_dart(self):
        dtrain, dtest = tm.load_agaricus(__file__)
        param = {
            "max_depth": 5,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "dart",
            "verbosity": 1,
        }
        # specify validations set to watch performance
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, watchlist)
        # this is prediction
        preds = bst.predict(dtest, iteration_range=(0, num_round))
        labels = dtest.get_label()
        err = sum(
            1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
        ) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            dtest_path = os.path.join(tmpdir, "dtest.dmatrix")
            model_path = os.path.join(tmpdir, "xgboost.model.dart.ubj")
            # save dmatrix into binary buffer
            dtest.save_binary(dtest_path)
            model_path = model_path
            # save model
            bst.save_model(model_path)
            # load model and data in
            bst2 = xgb.Booster(params=param, model_file=model_path)
            dtest2 = xgb.DMatrix(dtest_path)

        preds2 = bst2.predict(dtest2, iteration_range=(0, num_round))

        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

        def my_logloss(preds, dtrain):
            labels = dtrain.get_label()
            return "logloss", np.sum(np.log(np.where(labels, preds, 1 - preds)))

        # check whether custom evaluation metrics work
        bst = xgb.train(
            param, dtrain, num_round, evals=watchlist, custom_metric=my_logloss
        )
        preds3 = bst.predict(dtest, iteration_range=(0, num_round))
        assert all(preds3 == preds)

        # check whether sample_type and normalize_type work
        num_round = 50
        param["learning_rate"] = 0.1
        param["rate_drop"] = 0.1
        preds_list = []
        for p in [
            [p0, p1] for p0 in ["uniform", "weighted"] for p1 in ["tree", "forest"]
        ]:
            param["sample_type"] = p[0]
            param["normalize_type"] = p[1]
            bst = xgb.train(param, dtrain, num_round, evals=watchlist)
            preds = bst.predict(dtest, iteration_range=(0, num_round))
            err = sum(
                1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
            ) / float(len(preds))
            assert err < 0.1
            preds_list.append(preds)

        for ii in range(len(preds_list)):
            for jj in range(ii + 1, len(preds_list)):
                assert np.sum(np.abs(preds_list[ii] - preds_list[jj])) > 0

    def test_boost_from_prediction(self):
        # Re-construct dtrain here to avoid modification
        margined, _ = tm.load_agaricus(__file__)
        bst = xgb.train({"tree_method": "hist"}, margined, 1)
        predt_0 = bst.predict(margined, output_margin=True)
        margined.set_base_margin(predt_0)
        bst = xgb.train({"tree_method": "hist"}, margined, 1)
        predt_1 = bst.predict(margined)

        assert np.any(np.abs(predt_1 - predt_0) > 1e-6)
        dtrain, _ = tm.load_agaricus(__file__)
        bst = xgb.train({"tree_method": "hist"}, dtrain, 2)
        predt_2 = bst.predict(dtrain)
        assert np.all(np.abs(predt_2 - predt_1) < 1e-6)

    def test_boost_from_existing_model(self) -> None:
        X, _ = tm.load_agaricus(__file__)
        booster = xgb.train({"tree_method": "hist"}, X, num_boost_round=4)
        assert booster.num_boosted_rounds() == 4
        booster.set_param({"tree_method": "approx"})
        assert booster.num_boosted_rounds() == 4
        booster = xgb.train(
            {"tree_method": "hist"}, X, num_boost_round=4, xgb_model=booster
        )
        assert booster.num_boosted_rounds() == 8
        with pytest.warns(UserWarning, match="`updater`"):
            booster = xgb.train(
                {"updater": "prune", "process_type": "update"},
                X,
                num_boost_round=4,
                xgb_model=booster,
            )
        # Trees are moved for update, the rounds is reduced.  This test is
        # written for being compatible with current code (1.0.0).  If the
        # behaviour is considered sub-optimal, feel free to change.
        assert booster.num_boosted_rounds() == 4

        booster = xgb.train({"booster": "gblinear"}, X, num_boost_round=4)
        assert booster.num_boosted_rounds() == 4
        booster.set_param({"updater": "coord_descent"})
        assert booster.num_boosted_rounds() == 4
        booster.set_param({"updater": "shotgun"})
        assert booster.num_boosted_rounds() == 4
        booster = xgb.train(
            {"booster": "gblinear"}, X, num_boost_round=4, xgb_model=booster
        )
        assert booster.num_boosted_rounds() == 8

    def run_custom_objective(self, tree_method: Optional[str] = None):
        param = {
            "max_depth": 2,
            "eta": 1,
            "objective": "reg:logistic",
            "tree_method": tree_method,
        }
        dtrain, dtest = tm.load_agaricus(__file__)
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        num_round = 10

        def evalerror(preds: np.ndarray, dtrain: xgb.DMatrix):
            return tm.eval_error_metric(preds, dtrain, rev_link=True)

        # test custom_objective in training
        bst = xgb.train(
            param,
            dtrain,
            num_round,
            watchlist,
            obj=tm.logregobj,
            custom_metric=evalerror,
        )
        assert isinstance(bst, xgb.Booster)
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(
            1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
        ) / float(len(preds))
        assert err < 0.1

        # test custom_objective in cross-validation
        xgb.cv(
            param,
            dtrain,
            num_round,
            nfold=5,
            seed=0,
            obj=tm.logregobj,
            custom_metric=evalerror,
        )

        # test maximize parameter
        def neg_evalerror(preds, dtrain):
            labels = dtrain.get_label()
            preds = 1.0 / (1.0 + np.exp(-preds))
            return "error", float(sum(labels == (preds > 0.0))) / len(labels)

        bst2 = xgb.train(
            param,
            dtrain,
            num_round,
            evals=watchlist,
            obj=tm.logregobj,
            custom_metric=neg_evalerror,
            maximize=True,
        )
        preds2 = bst2.predict(dtest)
        err2 = sum(
            1 for i in range(len(preds2)) if int(preds2[i] > 0.5) != labels[i]
        ) / float(len(preds2))
        assert err == err2

    def test_custom_objective(self):
        self.run_custom_objective()

    def test_multi_eval_metric(self):
        dtrain, dtest = tm.load_agaricus(__file__)
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        param = {
            "max_depth": 2,
            "eta": 0.2,
            "verbosity": 1,
            "objective": "binary:logistic",
        }
        param["eval_metric"] = ["auc", "logloss", "error"]
        evals_result = {}
        bst = xgb.train(param, dtrain, 4, evals=watchlist, evals_result=evals_result)
        assert isinstance(bst, xgb.core.Booster)
        assert len(evals_result["eval"]) == 3
        assert set(evals_result["eval"].keys()) == {"auc", "error", "logloss"}

    def test_fpreproc(self):
        param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
        num_round = 2

        def fpreproc(dtrain, dtest, param):
            label = dtrain.get_label()
            ratio = float(np.sum(label == 0)) / np.sum(label == 1)
            param["scale_pos_weight"] = ratio
            return (dtrain, dtest, param)

        dtrain, _ = tm.load_agaricus(__file__)
        xgb.cv(
            param,
            dtrain,
            num_round,
            nfold=5,
            metrics={"auc"},
            seed=0,
            fpreproc=fpreproc,
        )

    def test_show_stdv(self):
        param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
        num_round = 2
        dtrain, _ = tm.load_agaricus(__file__)
        xgb.cv(
            param,
            dtrain,
            num_round,
            nfold=5,
            metrics={"error"},
            seed=0,
            show_stdv=False,
        )

    def test_prediction_cache(self) -> None:
        X, y = tm.make_sparse_regression(512, 4, 0.5, as_dense=False)
        Xy = xgb.DMatrix(X, y)
        param = {"max_depth": 8}
        booster = xgb.train(param, Xy, num_boost_round=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            booster.save_model(path)

            predt_0 = booster.predict(Xy)

            param["max_depth"] = 2

            booster = xgb.train(param, Xy, num_boost_round=1)
            predt_1 = booster.predict(Xy)
            assert not np.isclose(predt_0, predt_1).all()

            booster.load_model(path)
            predt_2 = booster.predict(Xy)
            np.testing.assert_allclose(predt_0, predt_2)

    def test_feature_names_validation(self):
        X = np.random.random((10, 3))
        y = np.random.randint(2, size=(10,))

        dm1 = xgb.DMatrix(X, y, feature_names=("a", "b", "c"))
        dm2 = xgb.DMatrix(X, y)

        bst = xgb.train([], dm1)
        bst.predict(dm1)  # success
        with pytest.raises(ValueError):
            bst.predict(dm2)
        bst.predict(dm1)  # success

        bst = xgb.train([], dm2)
        bst.predict(dm2)  # success

    @pytest.mark.skipif(**tm.no_json_schema())
    def test_json_dump_schema(self):
        import jsonschema

        def validate_model(parameters):
            X = np.random.random((100, 30))
            y = np.random.randint(0, 4, size=(100,))

            parameters["num_class"] = 4
            m = xgb.DMatrix(X, y)

            booster = xgb.train(parameters, m)
            dump = booster.get_dump(dump_format="json")

            for i in range(len(dump)):
                jsonschema.validate(instance=json.loads(dump[i]), schema=schema)

        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        doc = os.path.join(path, "doc", "dump.schema")
        with open(doc, "r") as fd:
            schema = json.load(fd)

        parameters = {
            "tree_method": "hist",
            "booster": "gbtree",
            "objective": "multi:softmax",
        }
        validate_model(parameters)

        parameters = {
            "tree_method": "hist",
            "booster": "dart",
            "objective": "multi:softmax",
        }
        validate_model(parameters)

    def test_special_model_dump_characters(self) -> None:
        params = {"objective": "reg:squarederror", "max_depth": 3}
        feature_names = ['"feature 0"', "\tfeature\n1", """feature "2"."""]
        X, y, w = tm.make_regression(n_samples=128, n_features=3, use_cupy=False)
        Xy = xgb.DMatrix(X, label=y, feature_names=feature_names)
        booster = xgb.train(params, Xy, num_boost_round=3)

        json_dump = booster.get_dump(dump_format="json")
        assert len(json_dump) == 3

        def validate_json(obj: dict) -> None:
            for k, v in obj.items():
                if k == "split":
                    assert v in feature_names
                elif isinstance(v, dict):
                    validate_json(v)

        for j_tree in json_dump:
            loaded = json.loads(j_tree)
            validate_json(loaded)

        dot_dump = booster.get_dump(dump_format="dot")
        for d in dot_dump:
            assert d.find(r"feature \"2\"") != -1

        text_dump = booster.get_dump(dump_format="text")
        for d in text_dump:
            assert d.find(r"feature \"2\"") != -1

    def run_slice(
        self,
        booster: xgb.Booster,
        dtrain: xgb.DMatrix,
        num_parallel_tree: int,
        num_classes: int,
        num_boost_round: int,
        use_np_type: bool,
    ):
        beg = 3
        if use_np_type:
            end: Integer = np.int32(7)
        else:
            end = 7

        sliced: xgb.Booster = booster[beg:end]
        assert sliced.feature_types == booster.feature_types

        sliced_trees = (end - beg) * num_parallel_tree * num_classes
        assert sliced_trees == len(sliced.get_dump())

        sliced_trees = sliced_trees // 2
        sliced = booster[beg:end:2]
        assert sliced_trees == len(sliced.get_dump())

        sliced = booster[beg:]
        sliced_trees = (num_boost_round - beg) * num_parallel_tree * num_classes
        assert sliced_trees == len(sliced.get_dump())

        sliced = booster[beg:]
        sliced_trees = (num_boost_round - beg) * num_parallel_tree * num_classes
        assert sliced_trees == len(sliced.get_dump())

        sliced = booster[:end]
        sliced_trees = end * num_parallel_tree * num_classes
        assert sliced_trees == len(sliced.get_dump())

        sliced = booster[:end]
        sliced_trees = end * num_parallel_tree * num_classes
        assert sliced_trees == len(sliced.get_dump())

        with pytest.raises(ValueError, match=r">= 0"):
            booster[-1:0]

        # we do not accept empty slice.
        with pytest.raises(ValueError, match="Empty slice"):
            booster[1:1]
        # stop can not be smaller than begin
        with pytest.raises(ValueError, match=r"Invalid.*"):
            booster[3:0]
        with pytest.raises(ValueError, match=r"Invalid.*"):
            booster[3:-1]
        # negative step is not supported.
        with pytest.raises(ValueError, match=r".*>= 1.*"):
            booster[0:2:-1]
        # step can not be 0.
        with pytest.raises(ValueError, match=r".*>= 1.*"):
            booster[0:2:0]

        trees = [_ for _ in booster]
        assert len(trees) == num_boost_round

        with pytest.raises(TypeError):
            booster["wrong type"]  # type: ignore
        with pytest.raises(IndexError):
            booster[: num_boost_round + 1]
        with pytest.raises(ValueError):
            booster[1, 2]  # too many dims
        # setitem is not implemented as model is immutable during slicing.
        with pytest.raises(TypeError):
            booster[:end] = booster  # type: ignore

        sliced_0 = booster[1:3]
        np.testing.assert_allclose(
            booster.predict(dtrain, iteration_range=(1, 3)), sliced_0.predict(dtrain)
        )
        sliced_1 = booster[3:7]
        np.testing.assert_allclose(
            booster.predict(dtrain, iteration_range=(3, 7)), sliced_1.predict(dtrain)
        )

        predt_0 = sliced_0.predict(dtrain, output_margin=True)
        predt_1 = sliced_1.predict(dtrain, output_margin=True)

        merged = predt_0 + predt_1 - 0.5  # base score.
        single = booster[1:7].predict(dtrain, output_margin=True)
        np.testing.assert_allclose(merged, single, atol=1e-6)

        sliced_0 = booster[1:7:2]  # 1,3,5
        sliced_1 = booster[2:8:2]  # 2,4,6

        predt_0 = sliced_0.predict(dtrain, output_margin=True)
        predt_1 = sliced_1.predict(dtrain, output_margin=True)

        merged = predt_0 + predt_1 - 0.5
        single = booster[1:7].predict(dtrain, output_margin=True)
        np.testing.assert_allclose(merged, single, atol=1e-6)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize("booster_name", ["gbtree", "dart"])
    def test_slice(self, booster_name: str) -> None:
        from sklearn.datasets import make_classification

        num_classes = 3
        X, y = make_classification(
            n_samples=1000, n_informative=5, n_classes=num_classes
        )
        dtrain = xgb.DMatrix(data=X, label=y)
        num_parallel_tree = 4
        num_boost_round = 16
        total_trees = num_parallel_tree * num_classes * num_boost_round
        booster = xgb.train(
            {
                "num_parallel_tree": num_parallel_tree,
                "subsample": 0.5,
                "num_class": num_classes,
                "booster": booster_name,
                "objective": "multi:softprob",
            },
            num_boost_round=num_boost_round,
            dtrain=dtrain,
        )
        booster.feature_types = ["q"] * X.shape[1]

        assert len(booster.get_dump()) == total_trees

        assert booster[...].num_boosted_rounds() == num_boost_round

        self.run_slice(
            booster, dtrain, num_parallel_tree, num_classes, num_boost_round, False
        )

        bytesarray = booster.save_raw(raw_format="ubj")
        booster = xgb.Booster(model_file=bytesarray)
        self.run_slice(
            booster, dtrain, num_parallel_tree, num_classes, num_boost_round, False
        )

        bytesarray = booster.save_raw(raw_format="deprecated")
        booster = xgb.Booster(model_file=bytesarray)
        self.run_slice(
            booster, dtrain, num_parallel_tree, num_classes, num_boost_round, True
        )

    def test_slice_multi(self) -> None:
        from sklearn.datasets import make_classification

        num_classes = 3
        X, y = make_classification(
            n_samples=1000, n_informative=5, n_classes=num_classes
        )
        Xy = xgb.DMatrix(data=X, label=y)
        num_parallel_tree = 4
        num_boost_round = 16

        booster = xgb.train(
            {
                "num_parallel_tree": num_parallel_tree,
                "num_class": num_classes,
                "booster": "gbtree",
                "objective": "multi:softprob",
                "multi_strategy": "multi_output_tree",
                "tree_method": "hist",
                "base_score": 0,
            },
            num_boost_round=num_boost_round,
            dtrain=Xy,
            callbacks=[ResetStrategy()],
        )
        sliced = [t for t in booster]
        assert len(sliced) == 16

        predt0 = booster.predict(Xy, output_margin=True)
        predt1 = np.zeros(predt0.shape)
        for t in booster:
            predt1 += t.predict(Xy, output_margin=True)

        np.testing.assert_allclose(predt0, predt1, atol=1e-5)

    @pytest.mark.skipif(**tm.no_pandas())
    @pytest.mark.parametrize("ext", ["json", "ubj"])
    def test_feature_info(self, ext: str) -> None:
        import pandas as pd

        # make data
        rows = 100
        cols = 10
        X = rng.randn(rows, cols)
        y = rng.randn(rows)

        # Test with pandas, which has feature info.
        feature_names = ["test_feature_" + str(i) for i in range(cols)]
        X_pd = pd.DataFrame(X, columns=feature_names)
        X_pd[f"test_feature_{3}"] = X_pd.iloc[:, 3].astype(np.int32)

        Xy = xgb.DMatrix(X_pd, y)
        assert Xy.feature_types is not None
        assert Xy.feature_types[3] == "int"
        booster = xgb.train({}, dtrain=Xy, num_boost_round=1)

        assert booster.feature_names == Xy.feature_names
        assert booster.feature_names == feature_names
        assert booster.feature_types == Xy.feature_types

        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + f"model.{ext}"
            booster.save_model(path)
            booster = xgb.Booster()
            booster.load_model(path)

            assert booster.feature_names == Xy.feature_names
            assert booster.feature_types == Xy.feature_types

        # Test with numpy, no feature info is set
        Xy = xgb.DMatrix(X, y)
        assert Xy.feature_names is None
        assert Xy.feature_types is None

        booster = xgb.train({}, dtrain=Xy, num_boost_round=1)
        assert booster.feature_names is None
        assert booster.feature_types is None

        # test explicitly set
        fns = [str(i) for i in range(cols)]
        booster.feature_names = fns

        assert booster.feature_names == fns

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"model.{ext}")
            booster.save_model(path)

            booster = xgb.Booster(model_file=path)
            assert booster.feature_names == fns
