"""Callback functions specialized for the Dask interface. The Dask interface accepts
normal training callbacks defined in :py:mod:`xgboost.callback` as well. Classes defined
here are tailored to the Dask interface.

    .. versionadded:: 3.0

"""

import logging

from distributed import Client

from ..callback import _EvaluationMonitorImpl

__all__ = ["EvaluationMonitor"]


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("[xgboost.dask]")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    return logger


class EvaluationMonitor(_EvaluationMonitorImpl):
    """Print the evaluation result at each iteration. The default monitor in the native
    interface logs the result to the Dask scheduler process. This class can be used to
    forward the logging to the client process. Important: see the `client` parameter for
    more info.

    .. versionadded:: 3.0.0

    """

    def __init__(
        self,
        client: Client,
        rank: int = 0,
        period: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        client :
            Distributed client. This must be the top-level client. The class uses
            :py:meth:`distributed.Client.forward_logging` in conjunction with the Python
            :py:mod:`logging` module to forward the evaluation results to the client
            process. It has undefined behaviour if called in a nested task. As a result,
            client-side logging is not enabled by default.

        Examples
        --------

        .. code-block:: python

            from distribtued import get_client

            from xgboost import dask as dxgb
            from xgboost.dask.callback import EvaluationMonitor


            def invalid():
                with get_client() as client:
                    EvaluationMonitor(client)  # Undefined!!

            def main(client):
                client.submit(invalid)  # bad

                callback = EvaluationMonitor(client)  # good
                # Disable the verbose_eval to use this custom monitor
                dxgb.train(client, {}, Xy, callbacks=[callback], verbose_eval=False)


            if __name__ == "__main__":
                with Client(scheduler_file="sched.json") as client:
                    main()

        """
        client.forward_logging(_get_logger().name)

        super().__init__(
            lambda msg: _get_logger().info(msg.strip()),
            rank=rank,
            period=period,
            show_stdv=False,
        )
