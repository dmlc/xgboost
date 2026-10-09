"""Temporary controlled probes for Intel/ARM CI performance.

Remove once the slow paths are explained. Keep wall/CPU timings separate from
cProfile: profiling Python calls and polling processes distorted earlier results.
Each configuration starts a fresh interpreter so native runtimes read its env.
"""

import argparse
import contextlib
import importlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

# All measurement callbacks run synchronously before their loop advances.
# ruff: noqa: B023

MODES = ("default", "xgb-one", "xgb-two", "blas-one", "passive")
ROOT = Path("macos-diagnostics/experiments")


def emit(**record):
    with (ROOT / f"{os.environ.get('MACOS_PROBE_MODE', 'driver')}.jsonl").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(record, default=str) + "\n")
    print(json.dumps(record, default=str), flush=True)


def measure(name, fn, repeat):
    wall, cpu = time.perf_counter(), time.process_time()
    result = fn()
    emit(
        phase=name,
        repeat=repeat,
        wall_seconds=time.perf_counter() - wall,
        # Parent CPU excludes child processes; worker startup has its own timing.
        process_cpu_seconds=time.process_time() - cpu,
    )
    return result


def worker_import(_):
    started = time.perf_counter()
    importlib.import_module("xgboost")
    importlib.import_module("pandas")
    return {"pid": os.getpid(), "import_seconds": time.perf_counter() - started}


def probe(mode, repeats):
    for package in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "distributed"):
        measure(f"import.{package}", lambda p=package: importlib.import_module(p), 0)

    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
    from threadpoolctl import threadpool_info, threadpool_limits
    from xgboost.testing.data import get_california_housing
    from xgboost.testing.ordinal import make_recoded

    threads = {"xgb-one": 1, "xgb-two": 2}.get(mode)
    limit = (
        threadpool_limits(limits=1, user_api="blas")
        if mode == "blas-one"
        else contextlib.nullcontext()
    )
    with limit:
        emit(
            event="environment",
            mode=mode,
            platform=platform.platform(),
            python=sys.version,
            cpus=os.cpu_count(),
            packages={
                name: importlib.import_module(name).__version__
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "sklearn",
                    "xgboost",
                    "distributed",
                    "loky",
                )
            },
            xgb_threads=threads,
            threadpools=threadpool_info(),
            environment={
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "OMP_WAIT_POLICY",
                    "KMP_BLOCKTIME",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        )
        X, y = measure("housing_data", get_california_housing, 0)
        grid = {"max_depth": [2, 4], "n_estimators": [50, 200]}
        for repeat in range(repeats):
            # Fixed data and folds reproduce the test's eight fits plus refit.
            # GridSearch stays serial; only estimator threading changes.
            for method in ("exact", "hist", "approx"):
                search = GridSearchCV(
                    xgb.XGBRegressor(
                        learning_rate=0.1, tree_method=method, n_jobs=threads
                    ),
                    grid,
                    cv=2,
                    n_jobs=1,
                )
                measure(f"grid.{method}", lambda: search.fit(X, y), repeat)
                emit(
                    event="grid_result",
                    method=method,
                    repeat=repeat,
                    best_params=search.best_params_,
                    best_score=search.best_score_,
                    mean_fit_time=search.cv_results_["mean_fit_time"].tolist(),
                    mean_score_time=search.cv_results_["mean_score_time"].tolist(),
                    refit_time=search.refit_time_,
                )

                # Same fitting/scoring workload without GridSearchCV orchestration.
                def direct_fits():
                    for params in ParameterGrid(grid):
                        for train, valid in KFold(2).split(X):
                            model = xgb.XGBRegressor(
                                learning_rate=0.1,
                                tree_method=method,
                                n_jobs=threads,
                                **params,
                            )
                            model.fit(X[train], y[train])
                            model.score(X[valid], y[valid])
                    model = xgb.XGBRegressor(
                        learning_rate=0.1,
                        tree_method=method,
                        n_jobs=threads,
                        **search.best_params_,
                    )
                    model.fit(X, y)
                    return model

                direct = measure(f"direct_fits.{method}", direct_fits, repeat)
                np.testing.assert_allclose(direct.predict(X), search.predict(X))

            enc, _, target, _, _ = measure(
                "categorical_data", lambda: make_recoded("cpu"), repeat
            )
            # Same values and categorical feature types, bypassing pandas input.
            array = measure(
                "pandas_to_numpy",
                lambda: np.column_stack(
                    [
                        enc[c].cat.codes.to_numpy()
                        if enc[c].dtype.name == "category"
                        else enc[c].to_numpy()
                        for c in enc.columns
                    ]
                ),
                repeat,
            )
            dm = measure(
                "dmatrix.pandas",
                lambda: xgb.DMatrix(
                    enc, target, enable_categorical=True, nthread=threads
                ),
                repeat,
            )
            dn = measure(
                "dmatrix.numpy",
                lambda: xgb.DMatrix(
                    array,
                    target,
                    feature_types=dm.feature_types,
                    feature_names=dm.feature_names,
                    nthread=threads,
                ),
                repeat,
            )
            params = {} if threads is None else {"nthread": threads}
            model = measure(
                "native_train", lambda: xgb.train(params, dm, num_boost_round=4), repeat
            )
            pred = measure("predict.pandas", lambda: model.inplace_predict(enc), repeat)
            cached = measure("predict.dmatrix", lambda: model.predict(dm), repeat)
            np.testing.assert_allclose(pred, cached)
            np.testing.assert_allclose(cached, model.predict(dn))
            # Controls for interpreter and BLAS performance, independent of XGBoost.
            measure(
                "python_loop", lambda: sum(i % 17 for i in range(1_000_000)), repeat
            )
            a = np.random.default_rng(2025).normal(size=(512, 512))
            measure("numpy_matmul", lambda: a @ a, repeat)

    if mode in ("default", "blas-one"):
        from loky import ProcessPoolExecutor

        for workers in (1, 4, 8):
            for repeat in range(repeats):

                def pool_round():
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        results = list(pool.map(worker_import, range(workers)))
                        emit(
                            event="worker_imports",
                            workers=workers,
                            repeat=repeat,
                            results=results,
                        )
                        measure(
                            f"pool.{workers}.warm_tasks",
                            lambda: list(pool.map(worker_import, range(workers))),
                            repeat,
                        )

                measure(f"pool.{workers}.startup_work_shutdown", pool_round, repeat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode:
        probe(args.mode, args.repeats)
        return
    failed = False
    for mode in MODES:
        env = dict(os.environ, MACOS_PROBE_MODE=mode)
        if mode == "passive":
            env.update(OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
        if mode == "blas-one":
            # Also applies to freshly spawned workers; parent additionally uses threadpoolctl.
            env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (ROOT / f"{mode}.log").open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--mode",
                        mode,
                        "--repeats",
                        str(args.repeats),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                returncode = process.wait(timeout=240)
                emit(event="completion", mode=mode, returncode=returncode)
                failed |= returncode != 0
            except subprocess.TimeoutExpired:
                # Kill worker descendants too, so a timed-out pool cannot affect later modes.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                emit(event="timeout", mode=mode, seconds=240)
                failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
