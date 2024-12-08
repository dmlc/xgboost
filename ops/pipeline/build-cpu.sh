#!/bin/bash

set -euox pipefail

echo "--- Build CPU code"

# This step is not necessary, but here we include it, to ensure that
# DMLC_CORE_USE_CMAKE flag is correctly propagated. We want to make sure that we use
# the configured header build/dmlc/build_config.h instead of
# include/dmlc/build_config_default.h.
rm -fv dmlc-core/include/dmlc/build_config_default.h

# Sanitizer tests
echo "--- Run Google Test with sanitizer enabled"
# Work around https://github.com/google/sanitizers/issues/1614
sudo sysctl vm.mmap_rnd_bits=28
python3 ops/docker_run.py \
  --container-id xgb-ci.cpu \
  -- ops/script/build_via_cmake.sh \
  -DUSE_SANITIZER=ON \
  -DENABLED_SANITIZERS="address;leak;undefined" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/
python3 ops/docker_run.py \
  --container-id xgb-ci.cpu \
  --run-args '-e ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer
  -e ASAN_OPTIONS=symbolize=1
  -e UBSAN_OPTIONS=print_stacktrace=1:log_path=ubsan_error.log
  --cap-add SYS_PTRACE' \
  -- bash -c \
  "cd build && ./testxgboost --gtest_filter=-*DeathTest*"

echo "--- Run Google Test"
python3 ops/docker_run.py \
  --container-id xgb-ci.cpu \
  -- ops/script/build_via_cmake.sh \
  -DCMAKE_PREFIX_PATH=/opt/grpc \
	-DPLUGIN_FEDERATED=ON
python3 ops/docker_run.py \
  --container-id xgb-ci.cpu \
  -- bash -c "cd build && ctest --extra-verbose"
