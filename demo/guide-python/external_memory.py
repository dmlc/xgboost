"""
Experimental support for external memory
========================================

This is similar to the one in `quantile_data_iterator.py`, but for external memory
instead of Quantile DMatrix.  The feature is not ready for production use yet.

    .. versionadded:: 1.5.0


See :doc:`the tutorial </tutorials/external_memory>` for more details.

"""
import os
import tempfile
from typing import Callable, List, Tuple

import numpy as np
from sklearn.datasets import make_regression

import xgboost


def make_batches(
    n_samples_per_batch: int, n_features: int, n_batches: int, tmpdir: str,
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
    def __init__(self, file_paths: List[Tuple[str, str]]):
        self._file_paths = file_paths
        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]
        X = np.load(X_path)
        y = np.load(y_path)
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> int:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the similar signature to
        # the ``DMatrix`` constructor.
        X, y = self.load_file()
        input_data(data=X, label=y)
        self._it += 1
        return 1

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0


def main(tmpdir: str) -> xgboost.Booster:
    # generate some random data for demo
    files = make_batches(1024, 17, 31, tmpdir)
    it = Iterator(files)
    # For non-data arguments, specify it here once instead of passing them by the `next`
    # method.
    missing = np.NaN
    Xy = xgboost.DMatrix(it, missing=missing, enable_categorical=False)

    # Other tree methods including ``hist`` and ``gpu_hist`` also work, see tutorial in
    # doc for details.
    booster = xgboost.train(
        {"tree_method": "approx", "max_depth": 2},
        Xy,
        evals=[(Xy, "Train")],
        num_boost_round=10,
    )
    return booster


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        main(tmpdir)
