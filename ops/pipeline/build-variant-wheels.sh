#!/bin/bash
## Build Python wheels using Wheel Variant prototype (WheelNext)
## This script runs inside a container (via GitHub Actions container support)
##
## Uses CUDA 12 build settings with RMM disabled.
## Does not modify the package name (keeps default 'xgboost').

set -euo pipefail

# Use the unified build script with explicit parameters:
# - CUDA 12
# - RMM disabled
exec bash ops/pipeline/build-cuda.sh \
  --cuda-version 12 \
  --use-rmm 0
