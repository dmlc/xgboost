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
- cuda-python
- pyhwloc

"""

import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
from functools import partial, update_wrapper, wraps
from typing import TYPE_CHECKING, Callable, List, ParamSpec, Tuple, TypeVar

import numpy as np
import xgboost
from loky import get_reusable_executor
from sklearn.datasets import make_regression
from xgboost import collective as coll
from xgboost.tracker import RabitTracker

if TYPE_CHECKING:
    from cuda.bindings.runtime import cudaError_t


def _checkcu(status: "cudaError_t") -> None:
    import cuda.bindings.runtime as cudart

    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(cudart.cudaGetErrorString(status))


def device_mem_total() -> int:
    """The total number of bytes of memory this GPU has."""
    import cuda.bindings.runtime as cudart

    status, _, total = cudart.cudaMemGetInfo()
    _checkcu(status)
    return total


def make_batches(
    n_samples_per_batch: int, n_features: int, n_batches: int, tmpdir: str, rank: int
) -> List[Tuple[str, str]]:
    """Create a single batch of data."""
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
            import cupy as cp

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


def setup_numa() -> None:
    """Set correct NUMA binding for GPU-based external memory training."""
    from pyhwloc import from_this_system
    from pyhwloc.cuda_runtime import get_device
    from pyhwloc.topology import MemBindFlags, MemBindPolicy, TypeFilter

    devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    assert devices is not None, "CUDA_VISIBLE_DEVICES must be set."

    with from_this_system().set_io_types_filter(TypeFilter.KEEP_ALL) as topo:
        # Get CPU affinity for this GPU. Device ordinal 0 is used because
        # CUDA_VISIBLE_DEVICES has already reordered the devices.
        dev = get_device(topo, device=0)
        cpuset = dev.get_affinity()

        # Set CPU binding
        topo.set_cpubind(cpuset)
        # Set memory binding with STRICT policy - ensures all memory allocations come
        # from the local NUMA node. hwloc determines the NUMA nodes from cpuset.
        topo.set_membind(cpuset, MemBindPolicy.BIND, MemBindFlags.STRICT)


def setup_async_pool() -> None:
    """Setup CUDA async pool. As an alternative, the RMM plugin can be used as well.
    This is the same as using the `CudaAsyncMemoryResource` from RMM, but without the
    RMM dependency.

    .. versionadded:: 3.2.0

    """
    import cuda.bindings.runtime as cudart
    from cuda.bindings import driver
    from cupy.cuda import MemoryAsyncPool

    status, dft_pool = cudart.cudaDeviceGetDefaultMemPool(0)
    _checkcu(status)

    total = device_mem_total()

    v = driver.cuuint64_t(int(total * 0.9))
    (status,) = cudart.cudaMemPoolSetAttribute(
        dft_pool,
        cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold,
        v,
    )
    _checkcu(status)
    # Set the allocator for cupy as well.
    import cupy as cp

    cp.cuda.set_allocator(MemoryAsyncPool().malloc)


R = TypeVar("R")
P = ParamSpec("P")


def try_run(fn: Callable[P, R]) -> Callable[P, R]:
    """Loky aborts the process without printing out any error message if there's an
    exception.

    """

    @wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            raise RuntimeError("Running into exception in worker.") from e

    return inner


@try_run
def hist_train(
    worker_idx: int,
    tmpdir: str,
    device: str,
    rabit_args: dict,
) -> None:
    """The hist tree method can use a special data structure `ExtMemQuantileDMatrix` for
    faster initialization and lower memory usage.

    """

    # Make sure XGBoost is using the configured memory pool for all allocations.
    with (
        coll.CommunicatorContext(**rabit_args),
        xgboost.config_context(
            use_cuda_async_pool=device == "cuda",
        ),
    ):
        print("Worker: ", worker_idx)
        # Generate the data for demonstration. The synthetic data is sharded by workers.
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
            # Check the first device
            assert (
                int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
                < coll.get_world_size()
            )
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


def launch_workers(tmpdir: str, args: argparse.Namespace) -> None:
    """Client function to launch workers."""
    n_workers = 2

    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
    tracker.start()
    rabit_args = tracker.worker_args()

    def initializer(device: str) -> None:
        # Set CUDA device before launching child processes.
        if device == "cuda":
            # name: LokyProcess-1
            _, sidx = mp.current_process().name.split("-")
            idx = int(sidx) - 1  # 1-based indexing from loky
            # Assuming two workers for demo.
            devices = ",".join([str(idx), str((idx + 1) % n_workers)])
            # P0: CUDA_VISIBLE_DEVICES=0,1
            # P1: CUDA_VISIBLE_DEVICES=1,0
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            setup_numa()
            setup_async_pool()

    with get_reusable_executor(
        max_workers=n_workers,
        initargs=(args.device,),
        initializer=initializer,
    ) as pool:
        # Poor man's currying
        fn = update_wrapper(
            partial(
                hist_train,
                tmpdir=tmpdir,
                device=args.device,
                rabit_args=rabit_args,
            ),
            hist_train,
        )
        pool.map(fn, range(n_workers))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        launch_workers(tmpdir, args)


if __name__ == "__main__":
    main()
