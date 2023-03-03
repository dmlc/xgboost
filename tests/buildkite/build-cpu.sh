#!/bin/bash

set -euo pipefail

echo "--- Build CPU code"

source tests/buildkite/conftest.sh

command_wrapper="tests/ci_build/ci_build.sh cpu docker"

$command_wrapper rm -fv dmlc-core/include/dmlc/build_config_default.h
  # This step is not necessary, but here we include it, to ensure that
  # DMLC_CORE_USE_CMAKE flag is correctly propagated. We want to make sure that we use
  # the configured header build/dmlc/build_config.h instead of
  # include/dmlc/build_config_default.h.
echo "--- Build libxgboost from the source"
$command_wrapper tests/ci_build/build_via_cmake.sh -DCMAKE_PREFIX_PATH=/opt/grpc \
  -DPLUGIN_DENSE_PARSER=ON -DPLUGIN_FEDERATED=ON
echo "--- Run Google Test"
$command_wrapper bash -c "cd build && ctest --extra-verbose"
echo "--- Stash XGBoost CLI executable"
buildkite-agent artifact upload ./xgboost

# Sanitizer test
echo "--- Run Google Test with sanitizer enabled"
$command_wrapper tests/ci_build/build_via_cmake.sh -DUSE_SANITIZER=ON \
  -DENABLED_SANITIZERS="address;leak;undefined" -DCMAKE_BUILD_TYPE=Debug \
  -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/
CI_DOCKER_EXTRA_PARAMS_INIT="-e ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer "`
  `"-e ASAN_OPTIONS=symbolize=1 "`
  `"-e UBSAN_OPTIONS=print_stacktrace=1:log_path=ubsan_error.log "`
  `"--cap-add SYS_PTRACE" \
  $command_wrapper bash -c "cd build && ctest --exclude-regex AllTestsInDMLCUnitTests "`
                           `"--extra-verbose"
