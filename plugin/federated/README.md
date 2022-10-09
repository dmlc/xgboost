XGBoost Plugin for Federated Learning
=====================================

This folder contains the plugin for federated learning. Follow these steps to build and test it.

Install gRPC
------------
We highly recommend installing gRPC in a local environment, such as a Conda environment,
by appropriately setting `CMAKE_INSTALL_PREFIX`.
There is no easy way to uninstall gRPC after you've installed it globally.

In the following example, we show how to build and install gRPC in a Conda environment.
```shell
sudo apt-get install build-essential autoconf libtool pkg-config cmake ninja-build
conda activate your_env
git clone -b v1.49.1 https://github.com/grpc/grpc \
    --recurse-submodules --depth 1 --shallow-submodules
cd grpc
cmake -S . -B build -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
cmake --build build --target install
cd ..
```

Build the Plugin
----------------
```shell
# Under xgboost source tree.
mkdir build
cd build
cmake .. -GNinja\
 -DPLUGIN_FEDERATED=ON\
 -DBUILD_WITH_CUDA_CUB=ON\
 -DUSE_CUDA=ON\
 -DUSE_NCCL=ON
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
