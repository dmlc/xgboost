# GPU Acceleration Demo

This demo shows how to perform a cross validation on the kaggle Bosch dataset with GPU acceleration. The Bosch numerical dataset has over 1 million rows and 968 features, making it time consuming to process.

This demo requires the [GPU plug-in](https://github.com/dmlc/xgboost/tree/master/plugin/updater_gpu) to be built and installed.

The dataset is available from:
https://www.kaggle.com/c/bosch-production-line-performance/data

Copy train_numeric.csv into xgboost/demo/data.

The subsample parameter can be changed so you can run the script first on a small portion of the data. Processing the entire dataset can take a long time and requires about 8GB of device memory. It is initially set to 0.4, using about 2650/3380MB on a GTX 970. 

```python
subsample = 0.4
```

Parameters are set as usual except that we set silent to 0 to see how much memory is being allocated on the GPU and we change 'updater' to 'grow_gpu' to activate the GPU plugin.

```python
param['silent'] = 0
param['updater'] = 'grow_gpu'
```

We use the sklearn cross validation function instead of the xgboost cv function as the xgboost cv will try to fit all folds in GPU memory at the same time.

Using the sklearn cv we can run each fold separately to fit a very large dataset onto the GPU.

Also note the line:
```python
del bst
```

This hints to the python garbage collection that it should delete the booster for the current fold before beginning the next. Without this line python may keep 'bst' from the previous fold in memory, using up precious GPU memory. 

You can change the updater parameter to run the equivalent algorithm for the CPU:
```python
param['updater'] = 'grow_colmaker'
```

Expect some minor variations in accuracy between the two versions.

