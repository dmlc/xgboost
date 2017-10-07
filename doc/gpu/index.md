XGBoost GPU Support
===================

This page contains information about GPU algorithms supported in XGBoost.
To install GPU support, checkout the [build and installation instructions](../build.md).

# CUDA Accelerated Tree Construction Algorithms
This plugin adds GPU accelerated tree construction and prediction algorithms to XGBoost.
## Usage
Specify the 'tree_method' parameter as one of the following algorithms. 

### Algorithms

```eval_rst
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| tree_method  | Description                                                                                                                                   |
+==============+===============================================================================================================================================+
| gpu_exact    | The standard XGBoost tree construction algorithm. Performs exact search for splits. Slower and uses considerably more memory than 'gpu_hist'  |
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| gpu_hist     | Equivalent to the XGBoost fast histogram algorithm. Faster and uses considerably less memory. Splits may be less accurate.                    |
+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
```

### Supported parameters 

```eval_rst
.. |tick| unicode:: U+2714 
.. |cross| unicode:: U+2718 

+--------------------+------------+-----------+
| parameter          | gpu_exact  | gpu_hist  |
+====================+============+===========+
| subsample          | |cross|    | |tick|    |
+--------------------+------------+-----------+
| colsample_bytree   | |cross|    | |tick|    |
+--------------------+------------+-----------+
| colsample_bylevel  | |cross|    | |tick|    |
+--------------------+------------+-----------+
| max_bin            | |cross|    | |tick|    |
+--------------------+------------+-----------+
| gpu_id             | |tick|     | |tick|    |
+--------------------+------------+-----------+
| n_gpus             | |cross|    | |tick|    |
+--------------------+------------+-----------+
| predictor          | |tick|     | |tick|    |
+--------------------+------------+-----------+

|  
```

GPU accelerated prediction is enabled by default for the above mentioned 'tree_method' parameters but can be switched to CPU prediction by setting 'predictor':'cpu_predictor'. This could be useful if you want to conserve GPU memory. Likewise when using CPU algorithms, GPU accelerated prediction can be enabled by setting 'predictor':'gpu_predictor'.

The device ordinal can be selected using the 'gpu_id' parameter, which defaults to 0.

Multiple GPUs can be used with the grow_gpu_hist parameter using the n_gpus parameter. which defaults to 1. If this is set to -1 all available GPUs will be used.  If gpu_id is specified as non-zero, the gpu device order is mod(gpu_id + i) % n_visible_devices for i=0 to n_gpus-1.  As with GPU vs. CPU, multi-GPU will not always be faster than a single GPU due to PCI bus bandwidth that can limit performance.  For example, when n_features * n_bins * 2^depth divided by time of each round/iteration becomes comparable to the real PCI 16x bus bandwidth of order 4GB/s to 10GB/s, then AllReduce will dominant code speed and multiple GPUs become ineffective at increasing performance.  Also, CPU overhead between GPU calls can limit usefulness of multiple GPUs.

This plugin currently works with the CLI version and python version.

Python example:
```python
param['gpu_id'] = 0
param['max_bin'] = 16
param['tree_method'] = 'gpu_hist'
```
## Benchmarks
To run benchmarks on synthetic data for binary classification:
```bash
$ python tests/benchmark/benchmark.py
```

Training time time on 1,000,000 rows x 50 columns with 500 boosting iterations and 0.25/0.75 test/train split on i7-6700K CPU @ 4.00GHz and Pascal Titan X.

```eval_rst
+--------------+----------+
| tree_method  | Time (s) |
+==============+==========+
| gpu_hist     | 13.87    |
+--------------+----------+
| hist         | 63.55    |
+--------------+----------+
| gpu_exact    | 161.08   |
+--------------+----------+
| exact        | 1082.20  |
+--------------+----------+

|  
```

[See here](http://dmlc.ml/2016/12/14/GPU-accelerated-xgboost.html) for additional performance benchmarks of the 'gpu_exact' tree_method.

## References
[Mitchell R, Frank E. (2017) Accelerating the XGBoost algorithm using GPU computing. PeerJ Computer Science 3:e127 https://doi.org/10.7717/peerj-cs.127](https://peerj.com/articles/cs-127/)

## Author
Rory Mitchell
Jonathan C. McKinney
Shankara Rao Thejaswi Nanditale
Vinay Deshpande
... and the rest of the H2O.ai and NVIDIA team.

Please report bugs to the xgboost/issues page.

