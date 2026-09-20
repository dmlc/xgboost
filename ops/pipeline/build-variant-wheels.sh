#!/bin/bash
## Build Python wheels using Wheel Variant prototype (WheelNext)
## This script runs inside a container (via GitHub Actions container support)
##
## Uses CUDA 13 build settings with RMM disabled.
## Does not modify the package name (keeps default 'xgboost').

set -euo pipefail

# Use the unified build script with explicit parameters:
# - CUDA 13
# - RMM disabled
exec bash ops/pipeline/build-cuda.sh \
  --cuda-version 13 \
  --use-rmm 0
