import numpy as np
import gc
import pytest
import xgboost as xgb
from hypothesis import given, strategies, assume, settings, note

import sys
import os

# sys.path.append("tests/python")
# import testing as tm
from xgboost import testing as tm

parameter_strategy = strategies.fixed_dictionaries(
    {
        "max_depth": strategies.integers(0, 11),
        "max_leaves": strategies.integers(0, 256),
        "max_bin": strategies.integers(2, 1024),
        "grow_policy": strategies.sampled_from(["lossguide", "depthwise"]),
        "single_precision_histogram": strategies.booleans(),
        "min_child_weight": strategies.floats(0.5, 2.0),
        "seed": strategies.integers(0, 10),
        # We cannot enable subsampling as the training loss can increase
        # 'subsample': strategies.floats(0.5, 1.0),
        "colsample_bytree": strategies.floats(0.5, 1.0),
        "colsample_bylevel": strategies.floats(0.5, 1.0),
    }
).filter(
    lambda x: (x["max_depth"] > 0 or x["max_leaves"] > 0)
    and (x["max_depth"] > 0 or x["grow_policy"] == "lossguide")
)


def train_result(param, dmat, num_rounds):
    result = {}
    xgb.train(
        param,
        dmat,
        num_rounds,
        [(dmat, "train")],
        verbose_eval=False,
        evals_result=result,
    )
    return result


class TestSYCLUpdaters:
    @given(parameter_strategy, strategies.integers(1, 5), tm.make_dataset_strategy())
    @settings(deadline=None)
    def test_sycl_hist(self, param, num_rounds, dataset):
        param["tree_method"] = "hist"
        param["device"] = "sycl"
        param["verbosity"] = 0
        param = dataset.set_params(param)
        result = train_result(param, dataset.get_dmat(), num_rounds)
        note(result)
        assert tm.non_increasing(result["train"][dataset.metric])

    @given(tm.make_dataset_strategy(), strategies.integers(0, 1))
    @settings(deadline=None)
    def test_specified_device_id_sycl_update(self, dataset, device_id):
        # Read the list of sycl-devicese
        sycl_ls = os.popen("sycl-ls").read()
        devices = sycl_ls.split("\n")

        # Test should launch only on gpu
        # Find gpus in the list of devices
        # and use the id in the list insteard of device_id
        target_device_type = "opencl:gpu"
        found_devices = 0
        for idx in range(len(devices)):
            if len(devices[idx]) >= len(target_device_type):
                if devices[idx][1 : 1 + len(target_device_type)] == target_device_type:
                    if found_devices == device_id:
                        param = {"device": f"sycl:gpu:{idx}"}
                        param = dataset.set_params(param)
                        result = train_result(param, dataset.get_dmat(), 10)
                        assert tm.non_increasing(result["train"][dataset.metric])
                    else:
                        found_devices += 1
