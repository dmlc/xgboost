"""Tests for the built-in objective Python interface."""

import json
from typing import TYPE_CHECKING, Dict

import numpy as np
import pytest

from ..core import DMatrix
from ..objective import (
    BinaryHinge,
    BinaryLogistic,
    BinaryLogitRaw,
    CountPoisson,
    MultiSoftmax,
    MultiSoftprob,
    RankMAP,
    RankNDCG,
    RankPairwise,
    RegAbsoluteError,
    RegExpectileError,
    RegGamma,
    RegLogistic,
    RegPseudoHuberError,
    RegQuantileError,
    RegSquaredError,
    RegSquaredLogError,
    RegTweedie,
    SurvivalAFT,
    SurvivalCox,
    _BuiltInObjective,
)
from ..sklearn import XGBClassifier
from ..training import train
from . import make_ltr, make_regression
from .data import get_cancer
from .utils import Device

if TYPE_CHECKING:
    from pytest import Subtests


def check_train_regression_objectives(device: Device) -> None:
    """Test training with regression objective classes."""
    X, y, _ = make_regression(100, 5, use_cupy=device == "cuda")
    dm = DMatrix(X, label=y)

    for obj_inst, obj_name in [
        (RegPseudoHuberError(delta=5.0), "reg:pseudohubererror"),
        (RegSquaredError(), "reg:squarederror"),
        (RegAbsoluteError(), "reg:absoluteerror"),
    ]:
        bst = train({"device": device}, dm, num_boost_round=5, obj=obj_inst)
        cfg = json.loads(bst.save_config())
        assert cfg["learner"]["objective"]["name"] == obj_name
        assert obj_inst.name == obj_name

    bst = train(
        {"device": device},
        dm,
        num_boost_round=5,
        obj=RegQuantileError(alpha=[0.1, 0.5, 0.9]),
    )
    pred = bst.predict(dm)
    assert pred.shape == (100, 3)
    assert RegQuantileError().name == "reg:quantileerror"

    bst = train(
        {"device": device},
        dm,
        num_boost_round=5,
        obj=RegExpectileError(alpha=[0.25, 0.75]),
    )
    pred = bst.predict(dm)
    assert pred.shape == (100, 2)
    assert RegExpectileError().name == "reg:expectileerror"


def check_train_positive_objectives(device: Device) -> None:
    """Test training with objectives requiring positive labels."""
    X, y, _ = make_regression(100, 5, use_cupy=device == "cuda")
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        y = cp.abs(y) + 0.1
    else:
        y = np.abs(y) + 0.1
    dm = DMatrix(X, label=y)

    for obj_inst, obj_name in [
        (RegTweedie(variance_power=1.8), "reg:tweedie"),
        (CountPoisson(max_delta_step=0.5), "count:poisson"),
        (RegGamma(), "reg:gamma"),
        (RegSquaredLogError(), "reg:squaredlogerror"),
    ]:
        bst = train({"device": device}, dm, num_boost_round=5, obj=obj_inst)
        cfg = json.loads(bst.save_config())
        assert cfg["learner"]["objective"]["name"] == obj_name
        assert obj_inst.name == obj_name


def check_train_classification_objectives(device: Device) -> None:
    """Test training with classification objective classes."""
    X, y = get_cancer()
    dm = DMatrix(X, label=y)

    for obj_inst, obj_name in [
        (RegLogistic(), "reg:logistic"),
        (BinaryLogistic(scale_pos_weight=2.0), "binary:logistic"),
        (BinaryLogitRaw(), "binary:logitraw"),
        (BinaryHinge(), "binary:hinge"),
    ]:
        bst = train({"device": device}, dm, num_boost_round=5, obj=obj_inst)
        cfg = json.loads(bst.save_config())
        assert cfg["learner"]["objective"]["name"] == obj_name
        assert obj_inst.name == obj_name

    datasets = pytest.importorskip("sklearn.datasets")
    X_mc, y_mc = datasets.load_digits(n_class=3, return_X_y=True)
    dm_mc = DMatrix(X_mc, label=y_mc)

    obj: _BuiltInObjective = MultiSoftmax(num_class=3)
    bst = train({"device": device}, dm_mc, num_boost_round=5, obj=obj)
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["name"] == "multi:softmax"
    assert obj.name == "multi:softmax"

    obj = MultiSoftprob(num_class=3)
    bst = train({"device": device}, dm_mc, num_boost_round=5, obj=obj)
    pred = bst.predict(dm_mc)
    assert pred.shape[1] == 3
    assert obj.name == "multi:softprob"


def check_train_survival_objectives(device: Device) -> None:
    """Test training with survival objective classes."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y_lower = np.abs(rng.randn(100))
    y_upper = y_lower + 1.0
    dm = DMatrix(X)
    dm.set_info(label_lower_bound=y_lower, label_upper_bound=y_upper)
    obj: _BuiltInObjective = SurvivalAFT(
        distribution="logistic", distribution_scale=2.0
    )
    bst = train({"device": device}, dm, num_boost_round=5, obj=obj)
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["name"] == "survival:aft"
    assert obj.name == "survival:aft"

    y_cox = np.abs(rng.randn(100)) + 0.1
    y_cox[:10] *= -1
    dm_cox = DMatrix(X, label=y_cox)
    obj = SurvivalCox()
    bst = train({"device": device}, dm_cox, num_boost_round=5, obj=obj)
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["name"] == "survival:cox"
    assert obj.name == "survival:cox"


def check_train_ranking_objectives(device: Device) -> None:
    """Test training with ranking objective classes."""
    X, y, qid, _ = make_ltr(100, 5, 4, max_rel=1)
    dm = DMatrix(X, label=y, qid=qid)

    for obj_inst, obj_name in [
        (RankNDCG(pair_method="mean", exp_gain=False), "rank:ndcg"),
        (RankPairwise(), "rank:pairwise"),
        (RankMAP(), "rank:map"),
    ]:
        bst = train({"device": device}, dm, num_boost_round=5, obj=obj_inst)
        cfg = json.loads(bst.save_config())
        assert cfg["learner"]["objective"]["name"] == obj_name
        assert obj_inst.name == obj_name


def check_equivalence(device: Device) -> None:
    """Test that class-based and string-based objectives produce identical results."""
    X, y, _ = make_regression(100, 5, use_cupy=False)
    dm = DMatrix(X, label=y)

    bst_cls = train(
        {"device": device}, dm, num_boost_round=10, obj=RegPseudoHuberError(delta=10.0)
    )
    bst_str = train(
        {"objective": "reg:pseudohubererror", "huber_slope": 10.0, "device": device},
        dm,
        num_boost_round=10,
    )
    np.testing.assert_allclose(bst_cls.predict(dm), bst_str.predict(dm), atol=1e-6)

    bst_cls = train(
        {"device": device},
        dm,
        num_boost_round=10,
        obj=RegQuantileError(alpha=[0.1, 0.5, 0.9]),
    )
    bst_str = train(
        {
            "objective": "reg:quantileerror",
            "quantile_alpha": "[0.1,0.5,0.9]",
            "device": device,
        },
        dm,
        num_boost_round=10,
    )
    np.testing.assert_allclose(bst_cls.predict(dm), bst_str.predict(dm), atol=1e-6)


def check_default_metrics(device: Device) -> None:
    """Test that built-in objectives set the correct default evaluation metrics."""
    X, y, _ = make_regression(100, 5, use_cupy=False)
    dm = DMatrix(X, label=y)

    result: Dict[str, Dict] = {}
    train(
        {"device": device},
        dm,
        num_boost_round=3,
        evals=[(dm, "train")],
        evals_result=result,
        obj=RegPseudoHuberError(delta=1.0),
        verbose_eval=False,
    )
    assert "mphe" in result["train"]

    result = {}
    train(
        {"device": device},
        dm,
        num_boost_round=3,
        evals=[(dm, "train")],
        evals_result=result,
        obj=RegQuantileError(alpha=[0.5]),
        verbose_eval=False,
    )
    assert "quantile" in result["train"]


def check_sklearn_objectives(device: Device) -> None:
    """Test objective classes with the scikit-learn interface."""
    X_bin, y_bin = get_cancer()
    clf = XGBClassifier(
        objective=BinaryLogistic(scale_pos_weight=2.0),
        n_estimators=5,
        device=device,
    )
    clf.fit(X_bin, y_bin)
    pred = clf.predict(X_bin)
    assert set(pred).issubset({0, 1})


def check_objectives(subtests: "Subtests", device: Device) -> None:
    """Run all tests."""
    for test in (
        check_default_metrics,
        check_equivalence,
        check_sklearn_objectives,
        check_train_classification_objectives,
        check_train_positive_objectives,
        check_train_ranking_objectives,
        check_train_regression_objectives,
        check_train_survival_objectives,
    ):
        with subtests.test(msg=test.__name__):
            test(device)
