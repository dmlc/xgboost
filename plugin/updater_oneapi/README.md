# DPC++-based Algorithm for Tree Construction
This plugin adds support of OneAPI programming model for tree construction and prediction algorithms to XGBoost.

## Usage
Specify the 'device_selector' parameter as one of the following options to offload model training on OneAPI device.
| Value | Description |
oneapi:cpu | Use default oneapi cpu |
oneapi:cpu:n | Use oneapi cpu with index n |
oneapi:gpu | Use default oneapi gpu |
oneapi:gpu:n | Use oneapi gpu with index n |
oneapi | Use default oneapi device |
oneapi:n | Use oneapi device with index n |
fit:oneapi:gpu; predict:cpu | Use oneapi gpu for fitting, and cpu for prediction

### Algorithms
| tree_method | Description |
| --- | --- |
hist | use hist method  |

### Algorithms
| objective | Description |
| --- | --- |
reg:squarederror | regression with squared loss  |
reg:squaredlogerror | regression with root mean squared logarithmic loss |
reg:logistic | logistic regression for probability regression task |
binary:logistic | logistic regression for binary classification task |
binary:logitraw | logistic regression for classification, output score before logistic transformation |
multi:softmax | multiclass classification using the softmax objective |
multi:softpred | multiclass classification using the softmax objective. Output is a vector of ndata * nclass|

Please note that parameter names are not finalized and can be changed during further integration of OneAPI support.

Python example:
```python
param['device_selector'] = 'oneapi:gpu'
param['tree_method'] = 'hist'
param['objective'] = 'reg:squarederror'
```

## Dependencies
Building the plugin requires Data Parallel C++ Compiler (https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html)

## Build
From the command line on Linux starting from the xgboost directory:

```bash
$ mkdir build
$ cd build
$ cmake .. -DPLUGIN_UPDATER_ONEAPI=ON -DINTEL_OMP_PATH=${CONDA_PREFIX}/lib
$ make -j
```
