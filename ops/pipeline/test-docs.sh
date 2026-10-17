#!/bin/bash
## Run doctests embedded in the Sphinx documentation.

set -euo pipefail

if [[ -z "${R_LIBS_USER:-}" ]]; then
  export R_LIBS_USER=/tmp/xgboost-r-doc-test-lib
fi

cmake_launcher_env=()
if command -v sccache >/dev/null 2>&1; then
  cmake_launcher_env=(
    CMAKE_C_COMPILER_LAUNCHER=sccache
    CMAKE_CXX_COMPILER_LAUNCHER=sccache
  )
fi

echo "--- Build libxgboost for documentation snippets"
cmake_args=(
  -S .
  -B build/doc-test
  -DHIDE_CXX_SYMBOLS=ON
  -DUSE_OPENMP=ON
  -DCMAKE_BUILD_TYPE=Release
)
env "${cmake_launcher_env[@]}" cmake "${cmake_args[@]}"
cmake --build build/doc-test --target xgboost --parallel "$(nproc)"

echo "--- Install R package for documentation snippets"
mkdir -p "${R_LIBS_USER}"
export R_LIBS_USER
MAKEFLAGS="-j$(nproc)" Rscript --vanilla R-package/tests/helper_scripts/install_deps.R doc-test
CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" R CMD INSTALL -l "${R_LIBS_USER}" R-package

echo "--- Run Sphinx doctest builder"
python3 -m sphinx -b doctest -a -E doc doc/_build/doctest
