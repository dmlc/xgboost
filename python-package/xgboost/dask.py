# pylint: disable=wrong-import-position,wrong-import-order,import-error
"""Dask extensions for xgboost"""
import os
import sys
import math
from threading import Thread
import distributed
import dask
import numpy as np
import pandas as pd
from . import rabit
from .core import DMatrix

# Try to find the dmlc tracker script

TRACKER_PATH = os.path.dirname(__file__) + "/../../dmlc-core/tracker/dmlc_tracker"
ALTERNATE_TRACKER_PATH = os.path.dirname(__file__) + "/../dmlc-core/tracker/dmlc_tracker"
sys.path.append(TRACKER_PATH)
sys.path.append(ALTERNATE_TRACKER_PATH)
from tracker import RabitTracker  # noqa


def __start_tracker():
    """ Start Rabit tracker """
    client = distributed.get_client()
    host = client.scheduler.address
    if '://' in host:
        host = host.rsplit('://', 1)[1]
    host = host.split(':')[0]
    n_workers = len(client.scheduler_info()['workers'])
    env = {'DMLC_NUM_WORKER': n_workers}
    rabit_context = RabitTracker(hostIP=host, nslave=n_workers)
    env.update(rabit_context.slave_envs())

    rabit_context.start(n_workers)
    thread = Thread(target=rabit_context.join)
    thread.daemon = True
    thread.start()
    return env


def __concat(data):
    if isinstance(data[0], np.ndarray):
        return np.concatenate(data, axis=0)
    if isinstance(data[0], (pd.DataFrame, pd.Series)):
        return pd.concat(data, axis=0)

    raise TypeError("Data must be either numpy arrays or pandas dataframes"
                    ". Got %s" % type(data[0]))


def create_worker_dmatrix(*args):
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
    for arg in args:
        if isinstance(arg, (dask.dataframe.core.DataFrame, dask.dataframe.core.Series)):
            if arg.npartitions != total_partitions:
                raise ValueError("Dask data must have the same partitions")
            # Get local partitions
            partitions = [arg.partitions[i].compute() for i in
                          range(begin_partition, end_partition)]
            if not partitions:
                raise ValueError("Worker " + str(
                    rabit.get_rank()) + " has no data. Try using smaller partitions")
            dmatrix_args.append(__concat(partitions))
    return DMatrix(dmatrix_args[0], *dmatrix_args[1:])


def __run_with_rabit(rabit_args, func, *args):
    os.environ["OMP_NUM_THREADS"] = str(distributed.get_worker().ncores)
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
    env = client.run_on_scheduler(__start_tracker)
    rabit_args = [('%s=%s' % item).encode() for item in env.items()]
    return client.run(__run_with_rabit, rabit_args, func, *args)
