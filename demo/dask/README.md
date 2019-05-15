# Dask Integration

[Dask](https://dask.org/) is a parallel computing library built on Python. Dask allows easy management of distributed workers and excels handling large distributed data science workflows.

This demo shows how to train and make predictions for an xgboost model on a distributed dask environment. We make use of first-class support in xgboost for launching dask workers. Workers launched in this manner are automatically connected via xgboosts underlying communication framework, Rabit. The calls to `xgb.train()` and `xgb.predict()` occur in parallel on each worker and are synchronized.

## Requirements
Dask is trivial to install from a conda environment. [See here for official install documentation](https://docs.dask.org/en/latest/install.html).

## Running the script
```bash
python dask_demo.py
```