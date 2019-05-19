# pylint: disable=wrong-import-position,wrong-import-order,import-error
"""Dask extensions for xgboost"""
import os
import sys
import math
from threading import Thread
from . import rabit
from .core import DMatrix
from .compat import (DaskDataFrame, DaskSeries, DaskArray,
                     distributed_get_worker)

# Try to find the dmlc tracker script

TRACKER_PATH = os.path.dirname(__file__) + "/../../dmlc-core/tracker/dmlc_tracker"
ALTERNATE_TRACKER_PATH = os.path.dirname(__file__) + "/dmlc-core/tracker/dmlc_tracker"
sys.path.append(TRACKER_PATH)
sys.path.append(ALTERNATE_TRACKER_PATH)
from tracker import RabitTracker  # noqa


def _start_tracker(host, n_workers):
    """ Start Rabit tracker """
    env = {'DMLC_NUM_WORKER': n_workers}
    rabit_context = RabitTracker(hostIP=host, nslave=n_workers)
    env.update(rabit_context.slave_envs())

    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def _merge_partitions(data, begin_partition, end_partition, total_partitions):
    if data.npartitions != total_partitions:
        raise ValueError("Dask data must have the same partitions")

    if isinstance(data, DaskArray):
        return data.blocks[begin_partition:end_partition].compute()

    return data.partitions[begin_partition:end_partition].compute()


def create_worker_dmatrix(*args, **kwargs):
    """
    Creates a DMatrix object local to a given worker. Simply forwards arguments onto the standard
    DMatrix constructor, if one of the arguments is a dask dataframe, unpack the data frame to
    get the local components.

    All dask dataframe arguments must use the same partitioning.

    :param args: DMatrix constructor args.
    :return: DMatrix object containing data local to current dask worker
    """
    total_partitions = args[0].npartitions
    partition_size = int(math.ceil(total_partitions / rabit.get_world_size()))
    begin_partition = partition_size * rabit.get_rank()
    end_partition = min(begin_partition + partition_size, total_partitions)
    dmatrix_args = []
    dmatrix_kwargs = {}
    # Convert positional args
    for arg in args:
        if isinstance(arg, (DaskDataFrame, DaskSeries, DaskArray)):
            dmatrix_args.append(
                _merge_partitions(arg, begin_partition, end_partition, total_partitions))
        else:
            dmatrix_args.append(arg)

    # Convert keyword args
    for k, v in kwargs.items():
        if isinstance(v, (DaskDataFrame, DaskSeries, DaskArray)):
            dmatrix_kwargs[k] = _merge_partitions(v, begin_partition, end_partition,
                                                  total_partitions)
        else:
            dmatrix_kwargs[k] = v

    return DMatrix(*dmatrix_args, **dmatrix_kwargs)


def _run_with_rabit(rabit_args, func, *args):
    os.environ["OMP_NUM_THREADS"] = str(distributed_get_worker().ncores)
    try:
        rabit.init(rabit_args)
        result = func(*args)
    finally:
        rabit.finalize()
    return result


def run(client, func, *args):
    """
    Launch arbitrary function on dask workers. Workers are connected by rabit, allowing
    distributed training. The environment variable OMP_NUM_THREADS is defined on each worker
    according to dask - this means that calls to xgb.train() will use the threads allocated by
    dask by default, unless the user overrides the nthread parameter.
    :param client: Dask client representing the cluster
    :param func: Python function to be executed by each worker. Typically contains xgboost
    training code.
    :param args: Arguments to be forwarded to func
    :return: Dict containing the function return value for each worker
    """
    host = client.scheduler.address
    if '://' in host:
        host = host.rsplit('://', 1)[1]
    host = host.split(':')[0]
    n_workers = len(client.scheduler_info()['workers'])
    env = client.run_on_scheduler(_start_tracker, host, n_workers)
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]
    return client.run(_run_with_rabit, rabit_args, func, *args)
