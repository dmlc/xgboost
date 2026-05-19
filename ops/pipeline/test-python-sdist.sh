#!/bin/bash
## Test installing Python XGBoost from source distribution

set -euox pipefail

# Stage the C++ source tree into python-package/cpp_src/ so the sdist is
# self-contained (scikit-build-core's sdist cannot reach files above
# python-package/ by itself).
python3 ops/script/prepare_sdist.py --clean

cd python-package
python --version
python -m build --sdist
pip install -v ./dist/xgboost-*.tar.gz
cd ..
python -c 'import xgboost'
