import json
import math
from pathlib import Path

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost._c_api import _parse_version
from xgboost.core import _parse_eval_str

dpath = "demo/data/"
rng = np.random.RandomState(1994)


class TestBasic:
    def test_compat(self):
        from xgboost.compat import lazy_isinstance

        a = np.array([1, 2, 3])
        assert lazy_isinstance(a, "numpy", "ndarray")
        assert not lazy_isinstance(a, "numpy", "dataframe")

    def test_basic(self, tmp_path: Path) -> None:
        dtrain, dtest = tm.load_agaricus(__file__)
        param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
        # specify validations set to watch performance
        watchlist = [(dtrain, "train")]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, evals=watchlist, verbose_eval=True)

        preds = bst.predict(dtrain)
        labels = dtrain.get_label()
        err = sum(
            1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
        ) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(
            1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]
        ) / float(len(preds))
        # error must be smaller than 10%
        assert err < 0.1

        dtest_path = tmp_path / "dtest.dmatrix"
        # save dmatrix into binary buffer
        dtest.save_binary(dtest_path)
        # save model
        model_path = tmp_path / "model.ubj"
        bst.save_model(model_path)
        # load model and data in
        bst2 = xgb.Booster(model_file=model_path)
        dtest2 = xgb.DMatrix(dtest_path)
        preds2 = bst2.predict(dtest2)
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

    def test_metric_config(self, tmp_path: Path) -> None:
        # Make sure that the metric configuration happens in booster so the string
        # `['error', 'auc']` doesn't get passed down to core.
        dtrain, dtest = tm.load_agaricus(__file__)
        param = {
            "max_depth": 2,
            "eta": 1,
            "objective": "binary:logistic",
            "eval_metric": ["error", "auc"],
        }
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        num_round = 2
        booster = xgb.train(param, dtrain, num_round, evals=watchlist)
        predt_0 = booster.predict(dtrain)
        path = tmp_path / "model.json"
        booster.save_model(path)

        booster = xgb.Booster(params=param, model_file=path)
        predt_1 = booster.predict(dtrain)
        np.testing.assert_allclose(predt_0, predt_1)

    def test_multiclass(self, tmp_path: Path) -> None:
        dtrain, dtest = tm.load_agaricus(__file__)
        param = {"max_depth": 2, "eta": 1, "num_class": 2}
        # specify validations set to watch performance
        watchlist = [(dtest, "eval"), (dtrain, "train")]
        num_round = 2
        bst = xgb.train(param, dtrain, num_round, evals=watchlist)
        # this is prediction
        preds = bst.predict(dtest)
        labels = dtest.get_label()
        err = sum(1 for i in range(len(preds)) if preds[i] != labels[i]) / float(
            len(preds)
        )
        # error must be smaller than 10%
        assert err < 0.1

        dtest_path = tmp_path / "dtest.buffer"
        model_path = tmp_path / "model.ubj"
        # save dmatrix into binary buffer
        dtest.save_binary(dtest_path)
        # save model
        bst.save_model(model_path)
        # load model and data in
        bst2 = xgb.Booster(model_file=model_path)
        dtest2 = xgb.DMatrix(dtest_path)
        preds2 = bst2.predict(dtest2)
        # assert they are the same
        assert np.sum(np.abs(preds2 - preds)) == 0

    @staticmethod
    def _check_dump(booster: xgb.Booster, vector_leaf: bool) -> None:
        text = booster.get_dump()
        assert len(text) == 1
        assert len(text[0].splitlines()) == 3
        assert "gain=" not in text[0]
        assert "cover=" not in text[0]

        text_stats = booster.get_dump(with_stats=True)
        assert len(text_stats[0].splitlines()) == 3
        assert "gain=" in text_stats[0]
        assert text_stats[0].count("cover=") == 3

        j_tree = json.loads(booster.get_dump(dump_format="json")[0])
        assert j_tree["nodeid"] == 0

        j_tree = json.loads(booster.get_dump(dump_format="json", with_stats=True)[0])
        assert isinstance(j_tree["gain"], (int, float))
        assert isinstance(j_tree["cover"], (int, float))
        for child in j_tree["children"]:
            assert isinstance(child["cover"], (int, float))
            assert isinstance(child["leaf"], list) == vector_leaf

        dot = booster.get_dump(dump_format="dot", with_stats=True)[0]
        assert "gain=" in dot
        assert dot.count("cover=") == 3

    def test_dump(self) -> None:
        data = np.arange(64, dtype=np.float32).reshape(32, 2)
        target = data[:, 0]
        features = ["Feature1", "Feature2"]

        dm = xgb.DMatrix(data, label=target, feature_names=features)
        params = {"eta": 0.3, "max_depth": 1, "tree_method": "hist"}

        booster = xgb.train(params, dm, num_boost_round=1)
        self._check_dump(booster, vector_leaf=False)

        y = np.column_stack((target, -target))
        dm = xgb.DMatrix(data, label=y, feature_names=features)
        params["multi_strategy"] = "multi_output_tree"
        vector_booster = xgb.train(params, dm, num_boost_round=1)
        self._check_dump(vector_booster, vector_leaf=True)

        with pytest.raises(ValueError):
            booster.get_dump(fmap="foo")

    @pytest.mark.skipif(**tm.no_pandas())
    @pytest.mark.parametrize("high", [3e10, 9.3e18], ids=["int64", "past-int64"])
    def test_dump_large_integer_split(self, high: float) -> None:
        # An integer split must not overflow in the dumps (#10035). Past int64 it has
        # no integer form and is written as a float.
        data = np.array([[0.0], [high]])
        dm = xgb.DMatrix(data, label=[0.0, 1.0], feature_types=["int"])
        params = {"max_depth": 1, "min_child_weight": 0, "eta": 1.0}
        booster = xgb.train(params, dm, num_boost_round=1)
        split = np.float32(booster.trees_to_dataframe()["Split"].dropna().item())
        cond = json.loads(booster.get_dump(dump_format="json")[0])["split_condition"]
        text = booster.get_dump(dump_format="text")[0]
        if split < 2.0**63:
            assert cond == math.ceil(split) > np.iinfo(np.int32).max
            assert f"<{cond}]" in text
        else:
            assert cond == pytest.approx(split)
            assert str(np.iinfo(np.int64).max) not in text

    def test_feature_score(self):
        rng = np.random.RandomState(0)
        data = rng.randn(100, 2)
        target = np.array([0, 1] * 50)
        features = ["F0"]
        with pytest.raises(ValueError):
            xgb.DMatrix(data, label=target, feature_names=features)

        params = {"objective": "binary:logistic"}
        dm = xgb.DMatrix(data, label=target, feature_names=["F0", "F1"])
        booster = xgb.train(params, dm, num_boost_round=1)
        # no error since feature names might be assigned before the booster seeing data
        # and booster doesn't known about the actual number of features.
        booster.feature_names = ["F0"]
        with pytest.raises(ValueError):
            booster.get_fscore()

        booster.feature_names = None
        # Use JSON to make sure the output has native Python type
        scores = json.loads(json.dumps(booster.get_fscore()))
        np.testing.assert_allclose(scores["f0"], 6.0)

    def test_load_file_invalid(self):
        with pytest.raises(xgb.core.XGBoostError):
            xgb.Booster(model_file="incorrect_path")

        with pytest.raises(xgb.core.XGBoostError):
            xgb.Booster(model_file="不正なパス")

    @pytest.mark.parametrize(
        "path", ["모델.ubj", "がうる・ぐら.json"], ids=["path-0", "path-1"]
    )
    def test_unicode_path(self, tmp_path: Path, path: str) -> None:
        model_path = tmp_path / path
        dtrain, _ = tm.load_agaricus(__file__)
        param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
        bst = xgb.train(param, dtrain, num_boost_round=2)
        bst.save_model(model_path)

        bst2 = xgb.Booster(model_file=model_path)
        assert bst.get_dump(dump_format="text") == bst2.get_dump(dump_format="text")

    def test_dmatrix_numpy_init_omp(self):
        rows = [1000, 11326, 15000]
        cols = 50
        for row in rows:
            X = np.random.randn(row, cols)
            y = np.random.randn(row).astype("f")
            dm = xgb.DMatrix(X, y, nthread=0)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

            dm = xgb.DMatrix(X, y, nthread=10)
            np.testing.assert_array_equal(dm.get_label(), y)
            assert dm.num_row() == row
            assert dm.num_col() == cols

    def test_cv(self):
        dm, _ = tm.load_agaricus(__file__)
        params = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, nfold=10, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_no_shuffle(self):
        dm, _ = tm.load_agaricus(__file__)
        params = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}

        # return np.ndarray
        cv = xgb.cv(
            params, dm, num_boost_round=10, shuffle=False, nfold=10, as_pandas=False
        )
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    @pytest.mark.skipif(**tm.no_sklearn())
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("seed", [0, 7])
    def test_cv_stratified_shuffle(self, shuffle: bool, seed: int) -> None:
        from sklearn.model_selection import StratifiedKFold

        X = np.arange(30, dtype=np.float32).reshape(-1, 1)
        y = np.arange(30) % 3
        dm = xgb.DMatrix(X, label=y)
        splitter = StratifiedKFold(
            n_splits=3, shuffle=shuffle, random_state=seed if shuffle else None
        )
        expected = list(splitter.split(X, y))
        actual = []

        def fpreproc(dtrain, dtest, params):
            actual.append(
                (
                    dtrain.get_data().toarray().ravel(),
                    dtest.get_data().toarray().ravel(),
                )
            )
            return dtrain, dtest, params

        xgb.cv(
            {"objective": "multi:softprob", "num_class": 3, "nthread": 1},
            dm,
            num_boost_round=1,
            nfold=3,
            stratified=True,
            shuffle=shuffle,
            seed=seed,
            fpreproc=fpreproc,
        )
        for (train, test), (expected_train, expected_test) in zip(
            actual, expected, strict=True
        ):
            np.testing.assert_array_equal(train, expected_train)
            np.testing.assert_array_equal(test, expected_test)

    def test_cv_explicit_fold_indices(self):
        dm, _ = tm.load_agaricus(__file__)
        params = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
        folds = [
            # Train        Test
            ([1, 3], [5, 8]),
            ([7, 9], [23, 43]),
        ]

        # return np.ndarray
        cv = xgb.cv(params, dm, num_boost_round=10, folds=folds, as_pandas=False)
        assert isinstance(cv, dict)
        assert len(cv) == (4)

    def test_cv_explicit_fold_indices_labels(self):
        params = {"max_depth": 2, "eta": 1, "objective": "reg:squarederror"}
        N = 100
        F = 3
        dm = xgb.DMatrix(data=np.random.randn(N, F), label=np.arange(N))
        folds = [
            # Train        Test
            ([1, 3], [5, 8]),
            ([7, 9], [23, 43, 11]),
        ]

        # Use callback to log the test labels in each fold
        class Callback(xgb.callback.TrainingCallback):
            def __init__(self) -> None:
                super().__init__()

            def after_iteration(
                self,
                model,
                epoch: int,
                evals_log: xgb.callback.TrainingCallback.EvalsLog,
            ):
                print([fold.dtest.get_label() for fold in model.cvfolds])

        cb = Callback()

        # Run cross validation and capture standard out to test callback result
        with tm.captured_output() as (out, err):
            xgb.cv(
                params,
                dm,
                num_boost_round=1,
                folds=folds,
                callbacks=[cb],
                as_pandas=False,
            )
            output = out.getvalue().strip()
        solution = (
            "[array([5., 8.], dtype=float32), array([23., 43., 11.],"
            + " dtype=float32)]"
        )
        assert output == solution


class TestBasicPathLike:
    """Unit tests using pathlib.Path for file interaction."""

    def test_DMatrix_init_from_path(self):
        """Initialization from the data path."""
        dtrain, _ = tm.load_agaricus(__file__)
        assert dtrain.num_row() == 6513
        assert dtrain.num_col() == 127

    def test_DMatrix_save_to_path(self, tmp_path: Path) -> None:
        """Saving to a binary file using pathlib from a DMatrix."""
        data = np.random.randn(100, 2)
        target = np.array([0, 1] * 50)
        features = ["Feature1", "Feature2"]

        dm = xgb.DMatrix(data, label=target, feature_names=features)

        binary_path = tmp_path / "dtrain.bin"
        dm.save_binary(binary_path)
        assert binary_path.exists()

    def test_Booster_init_invalid_path(self):
        """An invalid model_file path should raise XGBoostError."""
        with pytest.raises(xgb.core.XGBoostError):
            xgb.Booster(model_file=Path("invalidpath"))


def test_parse_ver() -> None:
    (major, minor, patch), post = _parse_version("2.1.0")
    assert post == ""
    (major, minor, patch), post = _parse_version("2.1.0-dev")
    assert post == "dev"
    (major, minor, patch), post = _parse_version("2.1.0rc1")
    assert post == "rc1"
    (major, minor, patch), post = _parse_version("2.1.0.post1")
    assert post == "post1"


@pytest.mark.parametrize(
    "data_name,metric_name",
    [
        ("train", "rmse"),
        ("fold:1", "rmse"),
        ("train", "my:rmse"),
        ("fold:1", "my:rmse"),
        ("fold:1:2", "my:custom:rmse"),
        ("検証:1", "誤差:rmse"),
    ],
)
@pytest.mark.parametrize(
    "score,expected",
    [
        ("0", 0.0),
        ("-0.0", -0.0),
        ("0.1234", 0.1234),
        ("-1.25", -1.25),
        ("+2.5", 2.5),
        ("1e-12", 1e-12),
        ("-2.5E+10", -2.5e10),
        ("inf", math.inf),
        ("+inf", math.inf),
        ("-inf", -math.inf),
        ("nan", math.nan),
        ("-nan", math.nan),
        # MSVC builds print indeterminate NaNs this way, see #6885.
        ("nan(ind)", math.nan),
        ("-nan(ind)", math.nan),
    ],
)
def test_parse_eval_str(
    data_name: str, metric_name: str, score: str, expected: float
) -> None:
    name = f"{data_name}-{metric_name}"
    result = _parse_eval_str(f"[42]\t{name}:{score}")
    assert len(result) == 1
    assert result[0][0] == name
    if math.isnan(expected):
        assert math.isnan(result[0][1])
    else:
        assert result[0][1] == expected
        if expected == 0:
            assert math.copysign(1, result[0][1]) == math.copysign(1, expected)


@pytest.mark.parametrize("separator", ["\t", " ", "  \t "])
def test_parse_eval_str_multiple_metrics(separator: str) -> None:
    result = _parse_eval_str(
        separator.join(
            [
                "[3]",
                "train-rmse:0.1234",
                "fold:1-my:rmse:2.5e-3",
                "eval-tweedie-nloglik:-inf",
                "fold:1-my:mae:1.5",
            ]
        )
        + "\n"
    )
    assert result == [
        ("train-rmse", 0.1234),
        ("fold:1-my:rmse", 0.0025),
        ("eval-tweedie-nloglik", -math.inf),
        ("fold:1-my:mae", 1.5),
    ]


@pytest.mark.parametrize("result", ["[0]", "[42]\t\n"])
def test_parse_eval_str_no_metrics(result: str) -> None:
    assert _parse_eval_str(result) == []


@pytest.mark.parametrize(
    "result", ["[0]\ttrain-rmse", "[0]\ttrain-rmse:", "[0]\tfold:1-my:rmse:bad"]
)
def test_parse_eval_str_invalid(result: str) -> None:
    with pytest.raises(ValueError):
        _parse_eval_str(result)
