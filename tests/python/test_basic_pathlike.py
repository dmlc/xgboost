# -*- coding: utf-8 -*-
import numpy as np
import pytest
import xgboost as xgb

from pathlib import Path

rng = np.random.RandomState(1994)

@pytest.fixture
def base_dpath():
    """Pathlib relative directory to testing data."""
    return Path('demo/data')


def test_DMatrix_init_from_path(base_dpath):
    """Initialization from the data path """
    dtrain = xgb.DMatrix(base_dpath / 'agaricus.txt.train')
    assert dtrain.num_row() == 6513
    assert dtrain.num_col() == 127


def test_DMatrix_save_to_path(tmp_path):
    """Saving to a binary file from a DMatrix works.
    Pytest built-in fixture tmp_path used for writing output location.
    """
    data = np.random.randn(100, 2)
    target = np.array([0, 1] * 50)
    features = ['Feature1', 'Feature2']

    dm = xgb.DMatrix(data, label=target, feature_names=features)

    binary_path = tmp_path / "train.bin"
    dm.save_binary(binary_path)
    assert binary_path.exists()


def test_Booster_init_invalid_path():
    """An invalid model_file should raise XGBoostError on init."""
    with pytest.raises(xgb.core.XGBoostError):
        bst = xgb.Booster(model_file=Path("invalidpath"))


def test_Booster_save_and_load(tmp_path):
    """Saving and loading model files from Paths.
    Pytest built-in fixture tmp_path used for writing output location.
    """
    save_path = tmp_path / "saveload.model"

    data = np.random.randn(100, 2)
    target = np.array([0, 1] * 50)
    features = ['Feature1', 'Feature2']

    dm = xgb.DMatrix(data, label=target, feature_names=features)
    params = {'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'eta': 0.3,
              'max_depth': 1}

    bst = xgb.train(params, dm, num_boost_round=1)

    # write to tmp_path
    bst.save_model(save_path)
    assert save_path.exists()

    def dump_assertions(dump):
        """Assertions for the expected dump from Booster"""
        assert len(dump) == 1, "Exepcted only 1 tree to be dumped."
        assert len(dump[0].splitlines()) == 3, "Expected 1 root and 2 leaves - 3 lines."

    # load the model again using Path
    bst2 = xgb.Booster(model_file=save_path)
    dump2 = bst2.get_dump()
    dump_assertions(dump2)

    # load again using load_model
    bst3 = xgb.Booster()
    bst3.load_model(save_path)
    dump3= bst3.get_dump()
    dump_assertions(dump3)

