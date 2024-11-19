"""
Experimental support for external memory
========================================

This is similar to the one in `quantile_data_iterator.py`, but for external memory
instead of Quantile DMatrix.  The feature is not ready for production use yet.

    .. versionadded:: 1.5.0


See :doc:`the tutorial </tutorials/external_memory>` for more details.

    .. versionchanged:: 3.0.0

        Added :py:class:`~xgboost.ExtMemQuantileDMatrix`.

To run the example, following packages in addition to XGBoost native dependencies are
required:

- scikit-learn

If `device` is `cuda`, following are also needed:

- cupy
- rmm
- python-cuda

"""

import argparse
import os
import tempfile
from typing import Callable, List, Tuple

import numpy as np
from sklearn.datasets import make_regression

import xgboost


def make_batches(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    tmpdir: str,
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    rng = np.random.RandomState(1994)
    for i in range(n_batches):
        X, y = make_regression(n_samples_per_batch, n_features, random_state=rng)
        X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
        y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
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


def hist_train(it: Iterator) -> None:
    """The hist tree method can use a special data structure `ExtMemQuantileDMatrix` for
    faster initialization and lower memory usage.

    .. versionadded:: 3.0.0

    """
    # For non-data arguments, specify it here once instead of passing them by the `next`
    # method.
    Xy = xgboost.ExtMemQuantileDMatrix(it, missing=np.nan, enable_categorical=False)
    booster = xgboost.train(
        {"tree_method": "hist", "max_depth": 4, "device": it.device},
        Xy,
        evals=[(Xy, "Train")],
        num_boost_round=10,
    )
    booster.predict(Xy)


def approx_train(it: Iterator) -> None:
    """The approx tree method uses the basic `DMatrix`."""

    # For non-data arguments, specify it here once instead of passing them by the `next`
    # method.
    Xy = xgboost.DMatrix(it, missing=np.nan, enable_categorical=False)
    # ``approx`` is also supported, but less efficient due to sketching. It's
    # recommended to use `hist` instead.
    booster = xgboost.train(
        {"tree_method": "approx", "max_depth": 4, "device": it.device},
        Xy,
        evals=[(Xy, "Train")],
        num_boost_round=10,
    )
    booster.predict(Xy)


def main(tmpdir: str, args: argparse.Namespace) -> None:
    """Entry point for training."""

    # generate some random data for demo
    files = make_batches(
        n_samples_per_batch=1024, n_features=17, n_batches=31, tmpdir=tmpdir
    )
    it = Iterator(args.device, files)

    hist_train(it)
    approx_train(it)


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
        # Make sure XGBoost is using RMM for all allocations.
        with xgboost.config_context(use_rmm=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                main(tmpdir, args)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            main(tmpdir, args)
