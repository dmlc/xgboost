#!/bin/bash
## Build and test XGBoost with AMD64 CPU

set -euo pipefail

source ops/pipeline/classify-git-branch.sh
source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

IMAGE_URI=${DOCKER_REGISTRY_URL}/xgb-ci.cpu:${IMAGE_TAG}

echo "--- Build CPU code"
set -x

# This step is not necessary, but here we include it, to ensure that
# DMLC_CORE_USE_CMAKE flag is correctly propagated. We want to make sure that we use
# the configured header build/dmlc/build_config.h instead of
# include/dmlc/build_config_default.h.
rm -fv dmlc-core/include/dmlc/build_config_default.h

# Test with sanitizer
export ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer
export ASAN_OPTIONS='symbolize=1'
export UBSAN_OPTIONS='print_stacktrace=1:log_path=ubsan_error.log'
# Work around https://github.com/google/sanitizers/issues/1614
sudo sysctl vm.mmap_rnd_bits=28
python3 ops/docker_run.py \
  --image-uri ${IMAGE_URI} \
  --run-args '-e ASAN_SYMBOLIZER_PATH -e ASAN_OPTIONS -e UBSAN_OPTIONS
    --cap-add SYS_PTRACE' \
  -- bash ops/pipeline/build-cpu-impl.sh cpu-sanitizer

# Test without sanitizer
rm -rf build/
python3 ops/docker_run.py \
  --image-uri ${IMAGE_URI} \
  -- bash ops/pipeline/build-cpu-impl.sh cpu
