#!/bin/bash
# Configure the XGBoost R package's CMake build to use sccache.
# This script creates ~/.R/Makevars with CMake compiler launchers.

set -euo pipefail

if [ -f ~/.R/Makevars ]; then
    echo "Error: ~/.R/Makevars already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p ~/.R
cat > ~/.R/Makevars << 'EOF'
XGBOOST_CMAKE_C_COMPILER_LAUNCHER = sccache
XGBOOST_CMAKE_CXX_COMPILER_LAUNCHER = sccache
XGBOOST_CMAKE_CUDA_COMPILER_LAUNCHER = sccache
EOF

echo "Configured the XGBoost R package to use sccache via ~/.R/Makevars"
