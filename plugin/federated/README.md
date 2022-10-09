XGBoost Plugin for Federated Learning
=====================================

This folder contains the plugin for federated learning. Follow these steps to build and test it.

Note. Building XGBoost with `-DPLUGIN_FEDERATED=ON` flag will automatically download the gRPC
source code and build it, along with its dependencies. This will increase compilation time.

Build the Plugin
----------------
```shell
# Under xgboost source tree.
mkdir build
cd build
# For now NCCL needs to be turned off.
cmake .. -GNinja\
 -DPLUGIN_FEDERATED=ON\
 -DUSE_CUDA=ON\
 -DBUILD_WITH_CUDA_CUB=ON\
 -DUSE_NCCL=OFF
ninja
cd ../python-package
pip install -e .  # or equivalently python setup.py develop
```

Test Federated XGBoost
----------------------
```shell
# Under xgboost source tree.
cd tests/distributed
# This tests both CPU training (`hist`) and GPU training (`gpu_hist`).
./runtests-federated.sh
```
