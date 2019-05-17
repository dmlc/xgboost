# Dask Integration

[Dask](https://dask.org/) is a parallel computing library built on Python. Dask allows easy management of distributed workers and excels handling large distributed data science workflows.

The simple demo shows how to train and make predictions for an xgboost model on a distributed dask environment. We make use of first-class support in xgboost for launching dask workers. Workers launched in this manner are automatically connected via xgboosts underlying communication framework, Rabit. The calls to `xgb.train()` and `xgb.predict()` occur in parallel on each worker and are synchronized.

The GPU demo shows how to configure and use GPUs on the local machine for training on a large dataset.

## Requirements
Dask is trivial to install using either pip or conda. [See here for official install documentation](https://docs.dask.org/en/latest/install.html).

The GPU demo requires [GPUtil](https://github.com/anderskm/gputil) for detecting system GPUs.

Install via `pip install gputil` 

## Running the scripts
```bash
python dask_simple_demo.py
python dask_gpu_demo.py
```