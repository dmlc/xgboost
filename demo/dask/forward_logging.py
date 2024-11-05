"""
Example of forwarding evaluation logs to the client
===================================================

The example runs on GPU. Two classes are defined to show how to use Dask builtins to
forward the logs to the client process.

"""

import logging

import dask
import distributed
from dask import array as da
from dask_cuda import LocalCUDACluster
from distributed import Client

from xgboost import dask as dxgb
from xgboost.callback import EvaluationMonitor


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("[xgboost.dask]")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    return logger


class ForwardLoggingMonitor(EvaluationMonitor):
    def __init__(
        self,
        client: Client,
        rank: int = 0,
        period: int = 1,
    ) -> None:
        """Print the evaluation result at each iteration. The default monitor in the
        native interface logs the result to the Dask scheduler process. This class can
        be used to forward the logging to the client process. Important: see the
        `client` parameter for more info.

        Parameters
        ----------
        client :
            Distributed client. This must be the top-level client. The class uses
            :py:meth:`distributed.Client.forward_logging` in conjunction with the Python
            :py:mod:`logging` module to forward the evaluation results to the client
            process. It has undefined behaviour if called in a nested task. As a result,
            client-side logging is not enabled by default.

        """
        client.forward_logging(_get_logger().name)

        super().__init__(
            rank=rank,
            period=period,
            logger=lambda msg: _get_logger().info(msg.strip()),
        )


class WorkerEventMonitor(EvaluationMonitor):
    """Use :py:meth:`distributed.print` to forward the log. A downside is that not only
    all clients connected to the cluster can see the log, the logs are also printed on
    the worker. If you use a local cluster, the log is duplicated.

    """

    def __init__(self, rank: int = 0, period: int = 1) -> None:
        super().__init__(
            rank=rank, period=period, logger=lambda msg: distributed.print(msg.strip())
        )


def hist_train(
    client: Client, X: da.Array, y: da.Array, monitor: EvaluationMonitor
) -> da.Array:
    # `DaskQuantileDMatrix` is used instead of `DaskDMatrix`, be careful that it can not
    # be used for anything else other than as a training DMatrix, unless a reference is
    # specified. See the `ref` argument of `DaskQuantileDMatrix`.
    dtrain = dxgb.DaskQuantileDMatrix(client, X, y)
    output = dxgb.train(
        client,
        # Make sure the device is set to CUDA.
        {"tree_method": "hist", "device": "cuda"},
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
        # Use the monitor to forward the log.
        callbacks=[monitor],
        # Disable the internal logging and prefer the client-side `EvaluationMonitor`.
        verbose_eval=False,
    )
    bst = output["booster"]
    history = output["history"]

    prediction = dxgb.predict(client, bst, X)
    print("Evaluation history:", history)
    return prediction


if __name__ == "__main__":
    # `LocalCUDACluster` is used for assigning GPU to XGBoost processes.  Here
    # `n_workers` represents the number of GPUs since we use one GPU per worker process.
    with LocalCUDACluster(n_workers=2, threads_per_worker=4) as cluster:
        # Create client from cluster, set the backend to GPU array (cupy).
        with Client(cluster) as client, dask.config.set({"array.backend": "cupy"}):
            # Generate some random data for demonstration
            rng = da.random.default_rng(1)

            m = 2**18
            n = 100
            X = rng.uniform(size=(m, n), chunks=(128**2, -1))
            y = X.sum(axis=1)

            # Use forwarding, the client must be the top client.
            monitor: EvaluationMonitor = ForwardLoggingMonitor(client)
            hist_train(client, X, y, monitor).compute()

            # Use distributed.print, the logs in this demo are duplicated as the same
            # log is printed in all workers along with the client.
            monitor = WorkerEventMonitor()
            hist_train(client, X, y, monitor).compute()
