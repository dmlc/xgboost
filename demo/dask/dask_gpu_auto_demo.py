from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import dask.array as da
import numpy as np
import xgboost as xgb
import GPUtil
import time


# Define the function to be executed on each worker
def train(X, y):
    dtrain = xgb.dask.create_worker_dmatrix(X, y)
    # Specify the GPU algorithm
    # If set GPU to be EXCLUSIVE mode by nvidia-smi -c EXCLUSIVE_PROCESS, native will
    # auto-allocate gpu for each worker
    params = {"tree_method": "gpu_hist"}
    print("Worker {} starting training on {} rows".format(xgb.rabit.get_rank(), dtrain.num_row()))
    start = time.time()
    xgb.train(params, dtrain, num_boost_round=500)
    end = time.time()
    print("Worker {} finished training in {:0.2f}s".format(xgb.rabit.get_rank(), end - start))


def main():
    max_devices = 16
    # Check which devices we have locally
    available_devices = GPUtil.getAvailable(limit=max_devices)
    # Use one worker per device
    cluster = LocalCluster(n_workers=len(available_devices), threads_per_worker=4)
    client = Client(cluster)

    # Set up a relatively large regression problem
    n = 100
    m = 10000000
    partition_size = 100000
    X = da.random.random((m, n), partition_size)
    y = da.random.random(m, partition_size)

    xgb.dask.run(client, train, X, y)


if __name__ == '__main__':
    main()
