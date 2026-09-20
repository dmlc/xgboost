#!/bin/bash
## Test installing Python XGBoost from source distribution

set -euox pipefail

# Stage the C++ source tree into python-package/cpp_src/ so the sdist is
# self-contained (scikit-build-core's sdist cannot reach files above
# python-package/ by itself).
python3 ops/script/prepare_sdist.py

cd python-package
python --version
python -m build --sdist
pip install -v ./dist/xgboost-*.tar.gz \
  --config-settings cmake.define.CMAKE_C_COMPILER_LAUNCHER=sccache \
  --config-settings cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=sccache
cd ..
python -c 'import xgboost'
