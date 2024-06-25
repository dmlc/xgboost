#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

command_wrapper="tests/ci_build/ci_build.sh aarch64"

# Build XGBoost4J binary
echo "--- Build libxgboost4j.so"
set -x
mkdir build
$command_wrapper bash -c \
  "cd build && cmake .. -GNinja -DJVM_BINDINGS=ON -DUSE_OPENMP=ON && ninja -v"
ldd lib/libxgboost4j.so

echo "--- Upload libxgboost4j.dylib"
pushd lib
libname=libxgboost4j_linux_arm64_${BUILDKITE_COMMIT}.so
mv -v libxgboost4j.so ${libname}
buildkite-agent artifact upload ${libname}
#if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
#then
  aws s3 cp ${libname} \
    s3://xgboost-nightly-builds/${BRANCH_NAME}/libxgboost4j/ \
    --acl public-read --no-progress
#fi
popd
