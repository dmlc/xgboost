"""Tests for the built-in objective Python interface."""

import json
from typing import Dict

import numpy as np
import pytest

from ..core import DMatrix
from ..objective import (
    AFT,
    MAP,
    NDCG,
    BinaryLogistic,
    ExpectileError,
    Gamma,
    Logistic,
    Objective,
    Pairwise,
    Poisson,
    PseudoHuber,
    QuantileError,
    Softmax,
    Softprob,
    Tweedie,
    _BuiltInObjective,
)
from ..sklearn import XGBClassifier, XGBRegressor
from ..training import train
from . import make_ltr, make_regression
from .data import get_cancer
from .utils import Device


def check_builtin_objective_base() -> None:
    """Test basic properties and error handling of built-in objective classes."""
    obj = PseudoHuber(delta=2.0)
    assert isinstance(obj, _BuiltInObjective)
    assert isinstance(obj, Objective)

    with pytest.raises(RuntimeError, match="computes gradients in C\\+\\+"):
        obj(0, np.array([1.0]), DMatrix(np.array([[1.0]])))

    with pytest.raises(TypeError, match="Unknown parameters"):
        PseudoHuber(bad_param=1.0)

    fp = PseudoHuber().flat_params()
    assert fp == {"objective": "reg:pseudohubererror"}

    fp = PseudoHuber(delta=10.0).flat_params()
    assert fp == {"objective": "reg:pseudohubererror", "huber_slope": "10.0"}

    fp = QuantileError(alpha=[0.1, 0.9]).flat_params()
    assert fp == {"objective": "reg:quantileerror", "quantile_alpha": "[0.1,0.9]"}

    fp = NDCG(unbiased=True, exp_gain=False).flat_params()
    assert fp["lambdarank_unbiased"] == "1"
    assert fp["ndcg_exp_gain"] == "0"


def check_train_regression_objectives(device: Device) -> None:
    """Test training with regression objective classes."""
    X, y, _ = make_regression(100, 5, use_cupy=device == "cuda")
    dm = DMatrix(X, label=y)

    bst = train({"device": device}, dm, num_boost_round=5, obj=PseudoHuber(delta=5.0))
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["name"] == "reg:pseudohubererror"
    assert PseudoHuber().name == "reg:pseudohubererror"

    bst = train(
        {"device": device},
        dm,
        num_boost_round=5,
        obj=QuantileError(alpha=[0.1, 0.5, 0.9]),
    )
    pred = bst.predict(dm)
    assert pred.shape == (100, 3)
    assert QuantileError().name == "reg:quantileerror"

    bst = train(
        {"device": device},
        dm,
        num_boost_round=5,
        obj=ExpectileError(alpha=[0.25, 0.75]),
    )
    pred = bst.predict(dm)
    assert pred.shape == (100, 2)
    assert ExpectileError().name == "reg:expectileerror"


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
        (Tweedie(variance_power=1.8), "reg:tweedie"),
        (Poisson(max_delta_step=0.5), "count:poisson"),
        (Gamma(), "reg:gamma"),
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
        (Logistic(), "reg:logistic"),
        (BinaryLogistic(scale_pos_weight=2.0), "binary:logistic"),
    ]:
        bst = train({"device": device}, dm, num_boost_round=5, obj=obj_inst)
        cfg = json.loads(bst.save_config())
        assert cfg["learner"]["objective"]["name"] == obj_name
        assert obj_inst.name == obj_name

    datasets = pytest.importorskip("sklearn.datasets")
    X_mc, y_mc = datasets.load_digits(n_class=3, return_X_y=True)
    dm_mc = DMatrix(X_mc, label=y_mc)

    obj = Softmax(num_class=3)
    bst = train({"device": device}, dm_mc, num_boost_round=5, obj=obj)
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["name"] == "multi:softmax"
    assert obj.name == "multi:softmax"

    obj = Softprob(num_class=3)
    bst = train({"device": device}, dm_mc, num_boost_round=5, obj=obj)
    pred = bst.predict(dm_mc)
    assert pred.shape[1] == 3
    assert obj.name == "multi:softprob"


def check_train_aft_objective(device: Device) -> None:
    """Test training with the AFT survival objective."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y_lower = np.abs(rng.randn(100))
    y_upper = y_lower + 1.0
    dm = DMatrix(X)
    dm.set_float_info("label_lower_bound", y_lower)
    dm.set_float_info("label_upper_bound", y_upper)

    obj = AFT(distribution="logistic", distribution_scale=2.0)
    bst = train({"device": device}, dm, num_boost_round=5, obj=obj)
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["name"] == "survival:aft"
    assert obj.name == "survival:aft"


def check_train_ranking_objectives(device: Device) -> None:
    """Test training with ranking objective classes."""
    X, y, qid, _ = make_ltr(100, 5, 4, max_rel=5)
    dm = DMatrix(X, label=y, qid=qid)

    for obj_inst, obj_name in [
        (NDCG(pair_method="mean", exp_gain=False), "rank:ndcg"),
        (Pairwise(), "rank:pairwise"),
    ]:
        bst = train({"device": device}, dm, num_boost_round=5, obj=obj_inst)
        cfg = json.loads(bst.save_config())
        assert cfg["learner"]["objective"]["name"] == obj_name
        assert obj_inst.name == obj_name

    y_bin = (y > np.median(y)).astype(np.float64)
    dm_bin = DMatrix(X, label=y_bin, qid=qid)
    obj = MAP()
    bst = train({"device": device}, dm_bin, num_boost_round=5, obj=obj)
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["name"] == "rank:map"
    assert obj.name == "rank:map"


def check_equivalence(device: Device) -> None:
    """Test that class-based and string-based objectives produce identical results."""
    X, y, _ = make_regression(100, 5, use_cupy=False)
    dm = DMatrix(X, label=y)

    bst_cls = train(
        {"device": device}, dm, num_boost_round=10, obj=PseudoHuber(delta=10.0)
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
        obj=QuantileError(alpha=[0.1, 0.5, 0.9]),
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
        obj=PseudoHuber(delta=1.0),
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
        obj=QuantileError(alpha=[0.5]),
        verbose_eval=False,
    )
    assert "quantile" in result["train"]


def check_sklearn_objectives(device: Device) -> None:
    """Test objective classes with the scikit-learn interface."""
    X, y, _ = make_regression(100, 5, use_cupy=False)

    reg = XGBRegressor(objective=PseudoHuber(delta=5.0), n_estimators=5, device=device)
    reg.fit(X, y)
    pred = reg.predict(X)
    assert pred.shape == (100,)
    assert isinstance(reg.objective, PseudoHuber)

    reg_cls = XGBRegressor(
        objective=PseudoHuber(delta=10.0), n_estimators=10, device=device
    )
    reg_cls.fit(X, y)

    reg_str = XGBRegressor(
        objective="reg:pseudohubererror",
        huber_slope=10.0,
        n_estimators=10,
        device=device,
    )
    reg_str.fit(X, y)

    np.testing.assert_allclose(reg_cls.predict(X), reg_str.predict(X), atol=1e-6)

    X_bin, y_bin = get_cancer()
    clf = XGBClassifier(
        objective=BinaryLogistic(scale_pos_weight=2.0),
        n_estimators=5,
        device=device,
    )
    clf.fit(X_bin, y_bin)
    pred = clf.predict(X_bin)
    assert set(pred).issubset({0, 1})
