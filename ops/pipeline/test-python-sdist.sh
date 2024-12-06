#!/bin/bash
## Test installing Python XGBoost from source distribution

set -euox pipefail

cd python-package
python --version
python -m build --sdist
pip install -v ./dist/xgboost-*.tar.gz
cd ..
python -c 'import xgboost'
