#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

command_wrapper="tests/ci_build/ci_build.sh jvm_manylinux2014_x86_64"

# Build XGBoost4J binary
echo "--- Build libxgboost4j.so (targeting glibc 2.17)"
set -x
mkdir build
$command_wrapper bash -c \
  "cd build && cmake .. -GNinja -DJVM_BINDINGS=ON -DUSE_OPENMP=ON && ninja -v"
ldd lib/libxgboost4j.so
objdump -T lib/libxgboost4j.so | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/\1/g' | sort -Vu

echo "--- Upload libxgboost4j.so"
pushd lib
libname=libxgboost4j_linux_x86_64_${BUILDKITE_COMMIT}.so
mv -v libxgboost4j.so ${libname}
buildkite-agent artifact upload ${libname}
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  aws s3 cp ${libname} \
    s3://xgboost-nightly-builds/${BRANCH_NAME}/libxgboost4j/ \
    --acl public-read --no-progress
fi
popd
