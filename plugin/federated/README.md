XGBoost Plugin for Federated Learning
=====================================

This folder contains the plugin for federated learning. Follow these steps to build and test it.

Install gRPC
------------
Refer to the [installation guide from the gRPC website](https://grpc.io/docs/languages/cpp/quickstart/).

Build the Plugin
----------------
```shell
# Under xgboost source tree.
mkdir build
cd build
cmake .. -GNinja \
 -DPLUGIN_FEDERATED=ON \
 -DUSE_CUDA=ON\
 -DUSE_NCCL=ON
ninja
cd ../python-package
pip install -e .
```
If CMake fails to locate gRPC, you may need to pass `-DCMAKE_PREFIX_PATH=<grpc path>` to CMake.

Test Federated XGBoost
----------------------
```shell
# Under xgboost source tree.
cd tests/distributed
# This tests both CPU training (`hist`) and GPU training (`gpu_hist`).
./runtests-federated.sh
```
