import os
from concurrent.futures import ThreadPoolExecutor
from typing import Type

import numpy as np
import pytest

import xgboost as xgb
from xgboost import testing as tm
from xgboost.testing.data import make_categorical
from xgboost.testing.ordinal import (
    run_cat_container,
    run_cat_container_iter,
    run_cat_container_mixed,
    run_cat_invalid,
    run_cat_leaf,
    run_cat_predict,
    run_cat_shap,
    run_cat_thread_safety,
    run_specified_cat,
)

pytestmark = pytest.mark.skipif(**tm.no_multiple(tm.no_arrow(), tm.no_cudf()))


def test_cat_container() -> None:
    run_cat_container("cuda")


def test_cat_container_mixed() -> None:
    run_cat_container_mixed("cuda")


def test_cat_container_iter() -> None:
    run_cat_container_iter("cuda")


def test_cat_predict() -> None:
    run_cat_predict("cuda")


def test_cat_invalid() -> None:
    run_cat_invalid("cuda")


def test_cat_thread_safety() -> None:
    run_cat_thread_safety("cuda")


def test_cat_shap() -> None:
    run_cat_shap("cuda")


def test_cat_leaf() -> None:
    run_cat_leaf("cuda")


def test_mixed_devices() -> None:
    n_samples = 128
    n_features = 4
    X, y = make_categorical(n_samples, n_features, 7, onehot=False, device="cpu")

    def run_cpu_gpu(DMatrixT: Type):
        Xy = DMatrixT(X, y, enable_categorical=True)
        booster = xgb.train({"tree_method": "hist", "device": "cuda"}, Xy)
        predt0 = booster.inplace_predict(X)
        predt1 = booster.predict(DMatrixT(X, y, enable_categorical=True))

        np.testing.assert_allclose(predt0, predt1)
        return True

    n_cpus = os.cpu_count()
    assert n_cpus is not None

    futures = []
    with ThreadPoolExecutor(max_workers=n_cpus) as e:
        for dm in (xgb.DMatrix, xgb.QuantileDMatrix):
            f = e.submit(run_cpu_gpu, dm)
            futures.append(f)

    for f in futures:
        assert f.result()

    X, y = make_categorical(n_samples, n_features, 7, onehot=False, device="cuda")

    def run_gpu_cpu(DMatrixT: Type):
        Xy = DMatrixT(X, y, enable_categorical=True)
        booster = xgb.train({"tree_method": "hist", "device": "cpu"}, Xy)
        predt0 = booster.inplace_predict(X).get()
        predt1 = booster.predict(DMatrixT(X, y, enable_categorical=True))

        np.testing.assert_allclose(predt0, predt1)
        return True

    futures = []
    with ThreadPoolExecutor(max_workers=n_cpus) as e:
        for dm in (xgb.DMatrix, xgb.QuantileDMatrix):
            f = e.submit(run_gpu_cpu, dm)
            futures.append(f)

    for f in futures:
        assert f.result()


def test_spcified_cat() -> None:
    run_specified_cat("cuda")
