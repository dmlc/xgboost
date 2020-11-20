# -*- coding: utf-8 -*-
import pytest
import xgboost as xgb


@pytest.mark.parametrize('verbosity_level', [0, 1, 2, 3])
def test_global_config_verbosity(verbosity_level):
    def get_current_verbosity():
        return xgb.get_config()['verbosity']

    old_verbosity = get_current_verbosity()
    with xgb.config_context(verbosity=verbosity_level):
        new_verbosity = get_current_verbosity()
        assert new_verbosity == verbosity_level
    assert old_verbosity == get_current_verbosity()
