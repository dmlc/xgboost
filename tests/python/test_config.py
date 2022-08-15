import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import pytest

import xgboost as xgb


@pytest.mark.parametrize("verbosity_level", [0, 1, 2, 3])
def test_global_config_verbosity(verbosity_level):
    def get_current_verbosity():
        return xgb.get_config()["verbosity"]

    old_verbosity = get_current_verbosity()
    with xgb.config_context(verbosity=verbosity_level):
        new_verbosity = get_current_verbosity()
        assert new_verbosity == verbosity_level
    assert old_verbosity == get_current_verbosity()


@pytest.mark.parametrize("use_rmm", [False, True])
def test_global_config_use_rmm(use_rmm):
    def get_current_use_rmm_flag():
        return xgb.get_config()["use_rmm"]

    old_use_rmm_flag = get_current_use_rmm_flag()
    with xgb.config_context(use_rmm=use_rmm):
        new_use_rmm_flag = get_current_use_rmm_flag()
        assert new_use_rmm_flag == use_rmm
    assert old_use_rmm_flag == get_current_use_rmm_flag()


def test_nested_config():
    with xgb.config_context(verbosity=3):
        assert xgb.get_config()["verbosity"] == 3
        with xgb.config_context(verbosity=2):
            assert xgb.get_config()["verbosity"] == 2
            with xgb.config_context(verbosity=1):
                assert xgb.get_config()["verbosity"] == 1
            assert xgb.get_config()["verbosity"] == 2
        assert xgb.get_config()["verbosity"] == 3

    with xgb.config_context(verbosity=3):
        assert xgb.get_config()["verbosity"] == 3
        with xgb.config_context(verbosity=None):
            assert xgb.get_config()["verbosity"] == 3  # None has no effect

    verbosity = xgb.get_config()["verbosity"]
    xgb.set_config(verbosity=2)
    assert xgb.get_config()["verbosity"] == 2
    with xgb.config_context(verbosity=3):
        assert xgb.get_config()["verbosity"] == 3
    xgb.set_config(verbosity=verbosity)  # reset


def test_thread_safty():
    n_threads = multiprocessing.cpu_count()
    futures = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i in range(256):
            f = executor.submit(test_nested_config)
            futures.append(f)

    for f in futures:
        f.result()
