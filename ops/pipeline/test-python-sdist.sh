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
sdist_source=(./dist/xgboost-*.tar.gz)
if [[ $(uname -s) == Linux ]]; then
  # pip unpacks archives into random paths, invalidating compiler cache entries.
  # Build the actual sdist contents at a stable path; never use the checkout's sources.
  sdist_dir="$PWD/../build-sdist"
  rm -rf "$sdist_dir"
  mkdir -p "$sdist_dir"
  tar -xzf "${sdist_source[0]}" --strip-components=1 -C "$sdist_dir"
  sdist_source=("$sdist_dir")
fi
pip install -v "${sdist_source[@]}" \
  --config-settings cmake.define.CMAKE_C_COMPILER_LAUNCHER=sccache \
  --config-settings cmake.define.CMAKE_CXX_COMPILER_LAUNCHER=sccache
cd ..
python -c 'import xgboost'
