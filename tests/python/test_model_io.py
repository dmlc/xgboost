import json
import locale
import os
import pickle
import tempfile

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm


def json_model(model_path: str, parameters: dict) -> dict:
    datasets = pytest.importorskip("sklearn.datasets")

    X, y = datasets.make_classification(64, n_features=8, n_classes=3, n_informative=6)
    if parameters.get("objective", None) == "multi:softmax":
        parameters["num_class"] = 3

    dm1 = xgb.DMatrix(X, y)

    bst = xgb.train(parameters, dm1)
    bst.save_model(model_path)

    if model_path.endswith("ubj"):
        import ubjson

        with open(model_path, "rb") as ubjfd:
            model = ubjson.load(ubjfd)
    else:
        with open(model_path, "r") as fd:
            model = json.load(fd)

    return model


class TestBoosterIO:
    def run_model_json_io(self, parameters: dict, ext: str) -> None:
        config = xgb.config.get_config()
        assert config["verbosity"] == 1

        if ext == "ubj" and tm.no_ubjson()["condition"]:
            pytest.skip(tm.no_ubjson()["reason"])

        loc = locale.getpreferredencoding(False)
        model_path = "test_model_json_io." + ext
        j_model = json_model(model_path, parameters)
        assert isinstance(j_model["learner"], dict)

        bst = xgb.Booster(model_file=model_path)

        bst.save_model(fname=model_path)
        if ext == "ubj":
            import ubjson

            with open(model_path, "rb") as ubjfd:
                j_model = ubjson.load(ubjfd)
        else:
            with open(model_path, "r") as fd:
                j_model = json.load(fd)

        assert isinstance(j_model["learner"], dict)

        os.remove(model_path)
        assert locale.getpreferredencoding(False) == loc

        json_raw = bst.save_raw(raw_format="json")
        from_jraw = xgb.Booster()
        from_jraw.load_model(json_raw)

        ubj_raw = bst.save_raw(raw_format="ubj")
        from_ubjraw = xgb.Booster()
        from_ubjraw.load_model(ubj_raw)

        if parameters.get("multi_strategy", None) != "multi_output_tree":
            # Old binary model is not supported for vector leaf.
            with pytest.warns(Warning, match="Model format is default to UBJSON"):
                old_from_json = from_jraw.save_raw(raw_format="deprecated")
                old_from_ubj = from_ubjraw.save_raw(raw_format="deprecated")

            assert old_from_json == old_from_ubj

        raw_json = bst.save_raw(raw_format="json")
        pretty = json.dumps(json.loads(raw_json), indent=2) + "\n\n"
        bst.load_model(bytearray(pretty, encoding="ascii"))

        if parameters.get("multi_strategy", None) != "multi_output_tree":
            # old binary model is not supported.
            with pytest.warns(Warning, match="Model format is default to UBJSON"):
                old_from_json = from_jraw.save_raw(raw_format="deprecated")
                old_from_ubj = from_ubjraw.save_raw(raw_format="deprecated")

            assert old_from_json == old_from_ubj

        rng = np.random.default_rng()
        X = rng.random(size=from_jraw.num_features() * 10).reshape(
            (10, from_jraw.num_features())
        )
        predt_from_jraw = from_jraw.predict(xgb.DMatrix(X))
        predt_from_bst = bst.predict(xgb.DMatrix(X))
        np.testing.assert_allclose(predt_from_jraw, predt_from_bst)

    @pytest.mark.parametrize("ext", ["json", "ubj"])
    def test_model_json_io(self, ext: str) -> None:
        parameters = {"booster": "gbtree", "tree_method": "hist"}
        self.run_model_json_io(parameters, ext)
        parameters = {
            "booster": "gbtree",
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree",
            "objective": "multi:softmax",
        }
        self.run_model_json_io(parameters, ext)
        parameters = {"booster": "gblinear"}
        self.run_model_json_io(parameters, ext)
        parameters = {"booster": "dart", "tree_method": "hist"}
        self.run_model_json_io(parameters, ext)

    def test_categorical_model_io(self) -> None:
        X, y = tm.make_categorical(256, 16, 71, False)
        Xy = xgb.DMatrix(X, y, enable_categorical=True)
        booster = xgb.train({"tree_method": "approx"}, Xy, num_boost_round=16)
        predt_0 = booster.predict(Xy)

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "model.deprecated")
            with pytest.raises(ValueError, match=r".*JSON/UBJSON.*"):
                with pytest.warns(Warning, match="Model format is default to UBJSON"):
                    booster.save_model(path)

            path = os.path.join(tempdir, "model.json")
            booster.save_model(path)
            booster = xgb.Booster(model_file=path)
            predt_1 = booster.predict(Xy)
            np.testing.assert_allclose(predt_0, predt_1)

            path = os.path.join(tempdir, "model.ubj")
            booster.save_model(path)
            booster = xgb.Booster(model_file=path)
            predt_1 = booster.predict(Xy)
            np.testing.assert_allclose(predt_0, predt_1)

    @pytest.mark.skipif(**tm.no_json_schema())
    def test_json_io_schema(self) -> None:
        import jsonschema

        model_path = "test_json_schema.json"
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        doc = os.path.join(path, "doc", "model.schema")
        with open(doc, "r") as fd:
            schema = json.load(fd)
        parameters = {"tree_method": "hist", "booster": "gbtree"}
        jsonschema.validate(instance=json_model(model_path, parameters), schema=schema)
        os.remove(model_path)

        parameters = {"tree_method": "hist", "booster": "dart"}
        jsonschema.validate(instance=json_model(model_path, parameters), schema=schema)
        os.remove(model_path)

        try:
            dtrain, _ = tm.load_agaricus(__file__)
            xgb.train({"objective": "foo"}, dtrain, num_boost_round=1)
        except ValueError as e:
            e_str = str(e)
            beg = e_str.find("Objective candidate")
            end = e_str.find("Stack trace")
            e_str = e_str[beg:end]
            e_str = e_str.strip()
            splited = e_str.splitlines()
            objectives = [s.split(": ")[1] for s in splited]
            j_objectives = schema["properties"]["learner"]["properties"]["objective"][
                "oneOf"
            ]
            objectives_from_schema = set()
            for j_obj in j_objectives:
                objectives_from_schema.add(j_obj["properties"]["name"]["const"])
            assert set(objectives) == objectives_from_schema

    def test_model_binary_io(self) -> None:
        model_path = "test_model_binary_io.deprecated"
        parameters = {
            "tree_method": "hist",
            "booster": "gbtree",
            "scale_pos_weight": "0.5",
        }
        X = np.random.random((10, 3))
        y = np.random.random((10,))
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(parameters, dtrain, num_boost_round=2)
        with pytest.warns(Warning, match="Model format is default to UBJSON"):
            bst.save_model(model_path)
        bst = xgb.Booster(model_file=model_path)
        os.remove(model_path)
        config = json.loads(bst.save_config())
        assert (
            float(config["learner"]["objective"]["reg_loss_param"]["scale_pos_weight"])
            == 0.5
        )

        buf = bst.save_raw()
        from_raw = xgb.Booster()
        from_raw.load_model(buf)

        buf_from_raw = from_raw.save_raw()
        assert buf == buf_from_raw


def save_load_model(model_path: str) -> None:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import KFold

    rng = np.random.RandomState(1994)

    digits = load_digits(n_class=2)
    y = digits["target"]
    X = digits["data"]
    kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(X, y):
        xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
        xgb_model.save_model(model_path)

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_path)

        assert isinstance(xgb_model.classes_, np.ndarray)
        np.testing.assert_equal(xgb_model.classes_, np.array([0, 1]))
        assert isinstance(xgb_model._Booster, xgb.Booster)

        preds = xgb_model.predict(X[test_index])
        labels = y[test_index]
        err = sum(
            1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
        ) / float(len(preds))
        assert err < 0.1
        assert xgb_model.get_booster().attr("scikit_learn") is None

        # test native booster
        preds = xgb_model.predict(X[test_index], output_margin=True)
        booster = xgb.Booster(model_file=model_path)
        predt_1 = booster.predict(xgb.DMatrix(X[test_index]), output_margin=True)
        assert np.allclose(preds, predt_1)

        with pytest.raises(TypeError):
            xgb_model = xgb.XGBModel()
            xgb_model.load_model(model_path)

    clf = xgb.XGBClassifier(booster="gblinear", early_stopping_rounds=1)
    clf.fit(X, y, eval_set=[(X, y)])
    best_iteration = clf.best_iteration
    best_score = clf.best_score
    predt_0 = clf.predict(X)
    clf.save_model(model_path)
    clf.load_model(model_path)
    assert clf.booster == "gblinear"
    predt_1 = clf.predict(X)
    np.testing.assert_allclose(predt_0, predt_1)
    assert clf.best_iteration == best_iteration
    assert clf.best_score == best_score

    clfpkl = pickle.dumps(clf)
    clf = pickle.loads(clfpkl)
    predt_2 = clf.predict(X)
    np.testing.assert_allclose(predt_0, predt_2)
    assert clf.best_iteration == best_iteration
    assert clf.best_score == best_score


@pytest.mark.skipif(**tm.no_sklearn())
def test_sklearn_model() -> None:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, "digits.deprecated")
        with pytest.warns(Warning, match="Model format is default to UBJSON"):
            save_load_model(model_path)

    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, "digits.model.json")
        save_load_model(model_path)

    with tempfile.TemporaryDirectory() as tempdir:
        model_path = os.path.join(tempdir, "digits.model.ubj")
        digits = load_digits(n_class=2)
        y = digits["target"]
        X = digits["data"]
        booster = xgb.train(
            {"tree_method": "hist", "objective": "binary:logistic"},
            dtrain=xgb.DMatrix(X, y),
            num_boost_round=4,
        )
        predt_0 = booster.predict(xgb.DMatrix(X))
        booster.save_model(model_path)
        cls = xgb.XGBClassifier()
        cls.load_model(model_path)

        proba = cls.predict_proba(X)
        assert proba.shape[0] == X.shape[0]
        assert proba.shape[1] == 2  # binary

        predt_1 = cls.predict_proba(X)[:, 1]
        assert np.allclose(predt_0, predt_1)

        cls = xgb.XGBModel()
        cls.load_model(model_path)
        predt_1 = cls.predict(X)
        assert np.allclose(predt_0, predt_1)

        # mclass
        X, y = load_digits(n_class=10, return_X_y=True)
        # small test_size to force early stop
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.01, random_state=1
        )
        clf = xgb.XGBClassifier(
            n_estimators=64, tree_method="hist", early_stopping_rounds=2
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        score = clf.best_score
        clf.save_model(model_path)

        clf = xgb.XGBClassifier()
        clf.load_model(model_path)
        assert clf.classes_.size == 10
        assert clf.objective == "multi:softprob"

        np.testing.assert_equal(clf.classes_, np.arange(10))
        assert clf.n_classes_ == 10

        assert clf.best_iteration == 27
        assert clf.best_score == score


@pytest.mark.skipif(**tm.no_sklearn())
def test_with_sklearn_obj_metric() -> None:
    from sklearn.metrics import mean_squared_error

    X, y = tm.datasets.make_regression()
    reg = xgb.XGBRegressor(objective=tm.ls_obj, eval_metric=mean_squared_error)
    reg.fit(X, y)

    pkl = pickle.dumps(reg)
    reg_1 = pickle.loads(pkl)
    assert callable(reg_1.objective)
    assert callable(reg_1.eval_metric)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.json")
        reg.save_model(path)

        reg_2 = xgb.XGBRegressor()
        reg_2.load_model(path)

    assert not callable(reg_2.objective)
    assert not callable(reg_2.eval_metric)
    assert reg_2.eval_metric is None
