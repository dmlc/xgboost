#!/bin/bash

set -euo pipefail

echo "--- Run clang-tidy"

source tests/buildkite/conftest.sh

tests/ci_build/ci_build.sh clang_tidy docker \
  --build-arg CUDA_VERSION_ARG=${CUDA_VERSION} \
  python3 tests/ci_build/tidy.py --cuda-archs 75
