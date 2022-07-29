import xgboost as xgb
import pytest


def use_rmm():
    use_rmm = xgb.build_info()["USE_RMM"]
    return {"condition": not use_rmm, "reason": "XGBoost is not compiled with RMM."}


@pytest.mark.skipif(**use_rmm())
@pytest.mark.parametrize('use_rmm', [False, True])
def test_global_config_use_rmm(use_rmm):
    def get_current_use_rmm_flag():
        return xgb.get_config()['use_rmm']

    old_use_rmm_flag = get_current_use_rmm_flag()
    with xgb.config_context(use_rmm=use_rmm):
        new_use_rmm_flag = get_current_use_rmm_flag()
        assert new_use_rmm_flag == use_rmm
    assert old_use_rmm_flag == get_current_use_rmm_flag()


def test_global_config_use_rmm_error():
    if not xgb.build_info()["USE_RMM"]:
        with pytest.raises(ValueError, match=r".*compiled.*"):
            xgb.set_config(use_rmm=True)
        # Check the value is not modified.
        assert xgb.get_config()["use_rmm"] is False
