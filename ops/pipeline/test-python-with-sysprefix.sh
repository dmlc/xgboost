#!/bin/bash
## Test if Python XGBoost can be configured to use libxgboost.so from the system prefix

set -euox pipefail

sudo apt-get update && sudo apt-get install -y ninja-build

mkdir build
pushd build
cmake .. -GNinja
ninja
popd

# Copy libxgboost.so to system prefix
cp -v lib/* "$(python -c 'import sys; print(sys.base_prefix)')/lib"

# Now configure Python XGBoost to use libxgboost.so from the system prefix
cd python-package
pip install virtualenv
virtualenv venv
source venv/bin/activate && \
  pip install -v . --config-settings use_system_libxgboost=True && \
  python -c 'import xgboost'
