# -*- coding: utf-8 -*-
import xgboost as xgb
import pytest
import testing as tm


@pytest.mark.parametrize('verbosity_level', [0, 1, 2, 3])
def test_global_config_verbosity(verbosity_level):
    def get_current_verbosity():
        return xgb.get_config()['verbosity']

    old_verbosity = get_current_verbosity()
    with xgb.config_context(verbosity=verbosity_level):
        new_verbosity = get_current_verbosity()
        assert new_verbosity == verbosity_level
    assert old_verbosity == get_current_verbosity()


@pytest.mark.parametrize('use_rmm', [False, True])
def test_global_config_use_rmm(use_rmm):
    def get_current_use_rmm_flag():
        return xgb.get_config()['use_rmm']

    old_use_rmm_flag = get_current_use_rmm_flag()
    with xgb.config_context(use_rmm=use_rmm):
        new_use_rmm_flag = get_current_use_rmm_flag()
        assert new_use_rmm_flag == use_rmm
    assert old_use_rmm_flag == get_current_use_rmm_flag()
