# DPC++-based Algorithm for Tree Construction
This plugin adds support of OneAPI programming model for tree construction and prediction algorithms to XGBoost.

## Usage
Specify the 'objective' parameter as one of the following options to offload computation of objective function on OneAPI device. 

### Algorithms
| objective | Description |
| --- | --- |
reg:squarederror_oneapi | regression with squared loss  |
reg:squaredlogerror_oneapi | regression with root mean squared logarithmic loss |
reg:logistic_oneapi | logistic regression for probability regression task |
binary:logistic_oneapi | logistic regression for binary classification task |
binary:logitraw_oneapi | logistic regression for classification, output score before logistic transformation |

Specify the 'predictor' parameter as one of the following options to offload prediction stage on OneAPI device. 

### Algorithms
| predictor | Description |
| --- | --- |
predictor_oneapi | prediction using OneAPI device  |

Please note that parameter names are not finalized and can be changed during further integration of OneAPI support.

Python example:
```python
param['predictor'] = 'predictor_oneapi'
param['objective'] = 'reg:squarederror_oneapi'
```

## Dependencies
Building the plugin requires Data Parallel C++ Compiler (https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/dpc-compiler.html)

## Build
From the command line on Linux starting from the xgboost directory:

```bash
$ mkdir build
$ cd build
$ EXPORT CXX=dpcpp && cmake .. -DPLUGIN_UPDATER_ONEAPI=ON
$ make -j
```
