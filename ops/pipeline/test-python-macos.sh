#!/bin/bash
## Test XGBoost Python wheel on MacOS

set -euox pipefail

brew install ninja

mkdir build
pushd build
# Set prefix, to use OpenMP library from Conda env
# See https://github.com/dmlc/xgboost/issues/7039#issuecomment-1025038228
# to learn why we don't use libomp from Homebrew.
cmake .. -GNinja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DBUILD_DEPRECATED_CLI=ON
ninja
popd

cd python-package
python --version
pip install -v .

cd ..
pytest -s -v -rxXs --durations=0 ./tests/python
pytest -s -v -rxXs --durations=0 ./tests/test_distributed/test_with_dask
