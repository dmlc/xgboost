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


def run_model_param_check(config: Dict[str, Any]) -> None:
    assert config["learner"]["learner_model_param"]["num_feature"] == str(4)
    assert config["learner"]["learner_train_param"]["booster"] == "gbtree"

    booster = config["learner"]["gradient_booster"]
    assert booster["name"] == "gbtree"
    assert booster["gbtree_model_param"]["num_parallel_tree"] == "2"


def run_booster_check(booster: xgboost.Booster, name: str) -> None:
    config = json.loads(booster.save_config())
    run_model_param_check(config)
    if name.find("cls") != -1:
        assert len(booster.get_dump()) == gm.kForests * gm.kRounds * gm.kClasses
        assert float(config["learner"]["learner_model_param"]["base_score"]) == 0.5
        assert config["learner"]["learner_train_param"]["objective"] == "multi:softmax"
    elif name.find("logitraw") != -1:
        assert len(booster.get_dump()) == gm.kForests * gm.kRounds
        assert config["learner"]["learner_model_param"]["num_class"] == str(0)
        assert (
            config["learner"]["learner_train_param"]["objective"] == "binary:logitraw"
        )
    elif name.find("logit") != -1:
        assert len(booster.get_dump()) == gm.kForests * gm.kRounds
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
        assert len(booster.get_dump()) == gm.kForests * gm.kRounds
        assert float(config["learner"]["learner_model_param"]["base_score"]) == 0.5
        assert (
            config["learner"]["learner_train_param"]["objective"] == "reg:squarederror"
        )


def run_scikit_model_check(name: str, path: str) -> None:
    if name.find("reg") != -1:
        reg = xgboost.XGBRegressor()
        reg.load_model(path)
        config = json.loads(reg.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "reg:squarederror"
        )
        assert len(reg.get_booster().get_dump()) == gm.kRounds * gm.kForests
        run_model_param_check(config)
    elif name.find("cls") != -1:
        cls = xgboost.XGBClassifier()
        cls.load_model(path)
        assert (
            len(cls.get_booster().get_dump()) == gm.kRounds * gm.kForests * gm.kClasses
        ), path
        config = json.loads(cls.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "multi:softprob"
        ), path
        run_model_param_check(config)
    elif name.find("ltr") != -1:
        ltr = xgboost.XGBRanker()
        ltr.load_model(path)
        assert len(ltr.get_booster().get_dump()) == gm.kRounds * gm.kForests
        config = json.loads(ltr.get_booster().save_config())
        assert config["learner"]["learner_train_param"]["objective"] == "rank:ndcg"
        run_model_param_check(config)
    elif name.find("logitraw") != -1:
        logit = xgboost.XGBClassifier()
        logit.load_model(path)
        assert len(logit.get_booster().get_dump()) == gm.kRounds * gm.kForests
        config = json.loads(logit.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "binary:logitraw"
        )
    elif name.find("logit") != -1:
        logit = xgboost.XGBClassifier()
        logit.load_model(path)
        assert len(logit.get_booster().get_dump()) == gm.kRounds * gm.kForests
        config = json.loads(logit.get_booster().save_config())
        assert (
            config["learner"]["learner_train_param"]["objective"] == "binary:logistic"
        )
    else:
        assert False


def download(path: str) -> None:
    """Download the model files from S3."""
    zip_path, _ = urllib.request.urlretrieve(
        "https://xgboost-ci-jenkins-artifacts.s3-us-west-2"
        + ".amazonaws.com/xgboost_model_compatibility_test.zip"
    )
    sha = "679b8ffd29c6ee4ebf8e9c0956064197f5d7c7a46d5b573b5794138bbe782aca"
    if hasattr(hashlib, "file_digest"):  # not in py 3.10
        with open(zip_path, "rb") as fd:
            digest = hashlib.file_digest(fd, "sha256")  # pylint: disable=attr-defined
            assert digest.hexdigest() == sha
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(path)


@pytest.mark.skipif(**tm.no_sklearn())
def test_model_compatibility() -> None:
    """Test model compatibility, can only be run on CI as others don't
    have the credentials.

    """
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "models")

    if not os.path.exists(path):
        download(path)

    models = [
        os.path.join(root, f)
        for root, subdir, files in os.walk(path)
        for f in files
        if f != "version"
    ]
    assert len(models) == 22

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
