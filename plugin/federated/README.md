XGBoost Plugin for Federated Learning
=====================================

This folder contains the plugin for federated learning.

Build
-----

The federated learning plugin requires `grpc` and `protobuf`. To install grpc, refer to the [installation guide from the gRPC website](https://grpc.io/docs/languages/cpp/quickstart/). Alternatively, one can use the `libgrpc` and the `protobuf` package from conda forge if conda is available. After obtaining the required dependencies, enable the flag: `-DPLUGIN_FEDERATED=ON` when running CMake. Please note that only Linux is supported for the federated plugin.

``` shell

cmake -B build -S . -DPLUGIN_FEDERATED=ON -GNinja
cd build && ninja

```

Test Federated XGBoost
----------------------

``` shell
# Under xgboost source tree.
pytest -sv tests/test_distributed/test_federated
```
