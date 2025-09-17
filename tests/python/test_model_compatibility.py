import copy
import hashlib
import json
import os
import urllib.request
import zipfile
from typing import Any, Dict

import generate_models as gm
import pytest

import xgboost
from xgboost import testing as tm
from xgboost.testing.updater import get_basescore


def run_model_param_check(name: str, config: Dict[str, Any]) -> None:
    assert config["learner"]["learner_model_param"]["num_feature"] == str(4)
    assert config["learner"]["learner_train_param"]["booster"] == "gbtree"

    booster = config["learner"]["gradient_booster"]
    assert booster["name"] == "gbtree"
    if name.find("1.0.0rc1") != -1:
        # There's no `num_parallel_tree` in the model parameter in 1.0 (it was a
        # configuration instead of a model parameter).
        return
    assert booster["gbtree_model_param"]["num_parallel_tree"] == str(gm.kForests)


def run_booster_check(booster: xgboost.Booster, name: str) -> None:
    config = json.loads(booster.save_config())
    run_model_param_check(name, config)
    n_rounds = get_n_rounds(name)
    if name.find("cls") != -1:
        assert len(booster.get_dump()) == gm.kForests * n_rounds * gm.kClasses
        base_score = get_basescore(config)
        assert isinstance(base_score, list)
        assert all(v == 0.5 for v in base_score)
        assert config["learner"]["learner_train_param"]["objective"] == "multi:softmax"
    elif name.find("logitraw") != -1:
        assert len(booster.get_dump()) == gm.kForests * n_rounds
        assert config["learner"]["learner_model_param"]["num_class"] == str(0)
        assert (
            config["learner"]["learner_train_param"]["objective"] == "binary:logitraw"
        )
    elif name.find("logit") != -1:
        assert len(booster.get_dump()) == gm.kForests * n_rounds
        assert config["learner"]["learner_model_param"]["num_class"] == str(0)
        assert (
            config["learner"]["learner_train_param"]["objective"] == "binary:logistic"
        )
    elif name.find("ltr") != -1:
        assert config["learner"]["learner_train_param"]["objective"] == "rank:ndcg"
    elif name.find("aft") != -1:
        assert config["learner"]["learner_train_param"]["objective"] == "survival:aft"
        assert (
            config["learner"]["objective"]["aft_loss_param"]["aft_loss_distribution"]
            == "normal"
        )
    else:
        assert name.find("reg") != -1
        assert len(booster.get_dump()) == gm.kForests * n_rounds
        assert get_basescore(config) == [0.5]
        assert (
            config["learner"]["learner_train_param"]["objective"] == "reg:squarederror"
        )


def get_n_rounds(name: str) -> int:
    if name.find("1.0.0rc1") != -1:
        n_rounds = 2
    else:
        n_rounds = gm.kRounds
    return n_rounds


def run_scikit_model_check(name: str, path: str) -> None:
    if name.find("reg") != -1:
        reg = xgboost.XGBRegressor()
        reg.load_model(path)
        config = json.loads(reg.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "reg:squarederror"
        )
        assert len(reg.get_booster().get_dump()) == get_n_rounds(name) * gm.kForests
        run_model_param_check(name, config)
    elif name.find("cls") != -1:
        cls = xgboost.XGBClassifier()
        cls.load_model(path)
        n_rounds = get_n_rounds(name)
        assert (
            len(cls.get_booster().get_dump()) == n_rounds * gm.kForests * gm.kClasses
        ), path
        config = json.loads(cls.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "multi:softprob"
        ), path
        run_model_param_check(name, config)
    elif name.find("ltr") != -1:
        ltr = xgboost.XGBRanker()
        ltr.load_model(path)
        assert len(ltr.get_booster().get_dump()) == get_n_rounds(name) * gm.kForests
        config = json.loads(ltr.get_booster().save_config())
        assert config["learner"]["learner_train_param"]["objective"] == "rank:ndcg"
        run_model_param_check(name, config)
    elif name.find("logitraw") != -1:
        logit = xgboost.XGBClassifier()
        logit.load_model(path)
        assert len(logit.get_booster().get_dump()) == get_n_rounds(name) * gm.kForests
        config = json.loads(logit.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "binary:logitraw"
        )
        run_model_param_check(name, config)
    elif name.find("logit") != -1:
        logit = xgboost.XGBClassifier()
        logit.load_model(path)
        assert len(logit.get_booster().get_dump()) == get_n_rounds(name) * gm.kForests
        config = json.loads(logit.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "binary:logistic"
        )
        run_model_param_check(name, config)
    else:
        assert False


def download(path: str) -> None:
    """Download the model files from S3."""
    zip_path, _ = urllib.request.urlretrieve(
        "https://xgboost-ci-jenkins-artifacts.s3-us-west-2"
        + ".amazonaws.com/xgboost_model_compatibility_tests-3.0.2.zip"
    )
    sha = "49d4d4db667a73590099dad9dca4f078532df05c5ea6e035ad4fa09596b1905a"
    if hasattr(hashlib, "file_digest"):  # not in py 3.10
        with open(zip_path, "rb") as fd:
            digest = hashlib.file_digest(fd, "sha256")  # pylint: disable=attr-defined
            assert digest.hexdigest() == sha
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)


@pytest.mark.skipif(**tm.no_sklearn())
def test_model_compatibility() -> None:
    """Test model compatibility."""
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "models")

    if not os.path.exists(path):
        download(path)

    models = [
        os.path.join(root, f) for root, subdir, files in os.walk(path) for f in files
    ]
    assert len(models) == 54

    for path in models:
        name = os.path.basename(path)
        if name.startswith("xgboost-"):
            booster = xgboost.Booster(model_file=path)
            run_booster_check(booster, name)
            # Do full serialization.
            booster = copy.copy(booster)
            run_booster_check(booster, name)
        elif name.startswith("xgboost_scikit"):
            run_scikit_model_check(name, path)
        else:
            assert False
