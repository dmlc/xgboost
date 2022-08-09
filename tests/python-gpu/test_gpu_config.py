import xgboost as xgb


def test_config():
    with xgb.config_context(device="CPU"):
        assert xgb.get_config()["device"] == "CPU"
        with xgb.config_context(device="CUDA"):
            assert xgb.get_config()["device"] == "CUDA"
