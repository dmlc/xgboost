# SYCL-based Algorithm for Tree Construction
This plugin adds support of SYCL programming model for prediction algorithms to XGBoost.

## Usage
Specify the 'device' parameter as described in the table below to offload model training and inference on SYCL device.

### Algorithms
| device | Description |
| --- | --- |
sycl | use default sycl device  |
sycl:gpu | use default sycl gpu  |
sycl:cpu | use default sycl cpu  |
sycl:gpu:N | use sycl gpu number N |
sycl:cpu:N | use sycl cpu number N |

Python example:
```python
param['device'] = 'sycl:gpu:0'
```

## Dependencies
Building the plugin requires Data Parallel C++ Compiler (https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)

## Build
From the command line on Linux starting from the xgboost directory:

```bash
$ mkdir build
$ cd build
$ cmake .. -DPLUGIN_SYCL=ON
$ make -j
```