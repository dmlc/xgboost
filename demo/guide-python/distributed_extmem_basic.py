"""
Experimental support for distributed training with external memory
==================================================================

    .. versionadded:: 3.0.0

See :doc:`the tutorial </tutorials/external_memory>` for more details. To run the
example, following packages in addition to XGBoost native dependencies are required:

- scikit-learn
- loky

If `device` is `cuda`, following are also needed:

- cupy
- python-cuda
- rmm

"""

import argparse
import multiprocessing as mp
import os
import tempfile
from functools import partial, update_wrapper
from typing import Callable, List, Tuple

import numpy as np
from loky import get_reusable_executor
from sklearn.datasets import make_regression

import xgboost
from xgboost import collective as coll
from xgboost.tracker import RabitTracker


def make_batches(
    n_samples_per_batch: int, n_features: int, n_batches: int, tmpdir: str, rank: int
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    rng = np.random.RandomState(rank)
    for i in range(n_batches):
        X, y = make_regression(n_samples_per_batch, n_features, random_state=rng)
        X_path = os.path.join(tmpdir, f"X-r{rank}-{i}.npy")
        y_path = os.path.join(tmpdir, f"y-r{rank}-{i}.npy")
        np.save(X_path, X)
        np.save(y_path, y)
        files.append((X_path, y_path))
    return files


class Iterator(xgboost.DataIter):
    """A custom iterator for loading files in batches."""

    def __init__(self, device: str, file_paths: List[Tuple[str, str]]) -> None:
        self.device = device

        self._file_paths = file_paths
        self._it = 0
        # XGBoost will generate some cache files under the current directory with the
        # prefix "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single batch of data."""
        X_path, y_path = self._file_paths[self._it]
        # When the `ExtMemQuantileDMatrix` is used, the device must match. GPU cannot
        # consume CPU input data and vice-versa.
        if self.device == "cpu":
            X = np.load(X_path)
            y = np.load(y_path)
        else:
            X = cp.load(X_path)
            y = cp.load(y_path)

        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> bool:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function
        is called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return False to let XGBoost know this is the end of iteration
            return False

        # input_data is a keyword-only function passed in by XGBoost and has the similar
        # signature to the ``DMatrix`` constructor.
        X, y = self.load_file()
        input_data(data=X, label=y)
        self._it += 1
        return True

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0


def setup_rmm() -> None:
    """Setup RMM for GPU-based external memory training."""
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    if not xgboost.build_info()["USE_RMM"]:
        return

    try:
        # Use the arena pool if available
        from cuda.bindings import runtime as cudart
        from rmm.mr import ArenaMemoryResource

        status, free, total = cudart.cudaMemGetInfo()
        if status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(cudart.cudaGetErrorString(status))

        mr = rmm.mr.CudaMemoryResource()
        mr = ArenaMemoryResource(mr, arena_size=int(total * 0.9))
    except ImportError:
        # The combination of pool and async is by design. As XGBoost needs to allocate
        # large pages repeatly, it's not easy to handle fragmentation. We can use more
        # experiments here.
        mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaAsyncMemoryResource())
        rmm.mr.set_current_device_resource(mr)
    # Set the allocator for cupy as well.
    cp.cuda.set_allocator(rmm_cupy_allocator)


def hist_train(worker_idx: int, tmpdir: str, device: str, rabit_args: dict) -> None:
    """The hist tree method can use a special data structure `ExtMemQuantileDMatrix` for
    faster initialization and lower memory usage.

    """

    # Make sure XGBoost is using RMM for all allocations.
    with coll.CommunicatorContext(**rabit_args), xgboost.config_context(use_rmm=True):
        # Generate the data for demonstration. The sythetic data is sharded by workers.
        files = make_batches(
            n_samples_per_batch=4096,
            n_features=16,
            n_batches=17,
            tmpdir=tmpdir,
            rank=coll.get_rank(),
        )
        # Since we are running two workers on a single node, we should divide the number
        # of threads between workers.
        n_threads = os.cpu_count()
        assert n_threads is not None
        n_threads = max(n_threads // coll.get_world_size(), 1)
        it = Iterator(device, files)
        Xy = xgboost.ExtMemQuantileDMatrix(
            it, missing=np.nan, enable_categorical=False, nthread=n_threads
        )
        # Check the device is correctly set.
        if device == "cuda":
            assert int(os.environ["CUDA_VISIBLE_DEVICES"]) < coll.get_world_size()
        booster = xgboost.train(
            {
                "tree_method": "hist",
                "max_depth": 4,
                "device": it.device,
                "nthread": n_threads,
            },
            Xy,
            evals=[(Xy, "Train")],
            num_boost_round=10,
        )
        booster.predict(Xy)


def main(tmpdir: str, args: argparse.Namespace) -> None:
    n_workers = 2

    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
    tracker.start()
    rabit_args = tracker.worker_args()

    def initializer(device: str) -> None:
        # Set CUDA device before launching child processes.
        if device == "cuda":
            # name: LokyProcess-1
            lop, sidx = mp.current_process().name.split("-")
            idx = int(sidx)  # 1-based indexing from loky
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx - 1)
            setup_rmm()

    with get_reusable_executor(
        max_workers=n_workers, initargs=(args.device,), initializer=initializer
    ) as pool:
        # Poor man's currying
        fn = update_wrapper(
            partial(
                hist_train, tmpdir=tmpdir, device=args.device, rabit_args=rabit_args
            ),
            hist_train,
        )
        pool.map(fn, range(n_workers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        import cupy as cp

        # It's important to use RMM with `CudaAsyncMemoryResource`. for GPU-based
        # external memory to improve performance. If XGBoost is not built with RMM
        # support, a warning is raised when constructing the `DMatrix`.
        setup_rmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            main(tmpdir, args)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            main(tmpdir, args)
