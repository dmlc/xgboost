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


@pytest.mark.skipif(**tm.no_dask())
def test_global_config_with_dask():
    from distributed import Client, LocalCluster

    xgb.set_config(verbosity=0)
    config = xgb.get_config()
    assert config['verbosity'] == 0
    with LocalCluster(n_workers=4) as cluster:
        with Client(cluster) as client:
            # By default, the global configuration of the scheduler process is not shared with the
            # the worker processes.
            for config in client.run(xgb.get_config).values():
                assert config['verbosity'] != 0
            # Use client.run() to configure global configuration for the worker processes:
            client.run(xgb.set_config, verbosity=0)
            for config in client.run(xgb.get_config).values():
                assert config['verbosity'] == 0
