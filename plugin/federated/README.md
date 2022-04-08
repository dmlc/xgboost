XGBoost Plugin for Federated Learning
=====================================

This folder contains the plugin for federated learning.

Install gRPC
------------
```shell
sudo apt-get install build-essential autoconf libtool pkg-config cmake ninja-build
git clone -b v1.45.1 https://github.com/grpc/grpc
cd grpc
git submodule update --init
cmake -S . -B build -GNinja\
 -DABSL_PROPAGATE_CXX_STD=ON
cmake --build build --target install
```
