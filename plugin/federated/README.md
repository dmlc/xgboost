XGBoost Plugin for Federated Learning
=====================================

This folder contains the plugin for federated learning.

See [build instruction](../../doc/build.rst) for how to build the plugin.


Test Federated XGBoost
----------------------
```shell
# Under xgboost source tree.
cd tests/distributed/test_federated
# This tests both CPU training (`hist`) and GPU training (`gpu_hist`).
./runtests-federated.sh
```
