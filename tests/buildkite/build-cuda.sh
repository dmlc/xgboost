#!/bin/bash

set -euo pipefail

CUDA_VERSION=11.0.3
WHEEL_TAG=manylinux2014_x86_64

echo "--- Build with CUDA ${CUDA_VERSION}"

source tests/buildkite/conftest.sh

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

command_wrapper="tests/ci_build/ci_build.sh gpu_build_centos7 docker --build-arg "`
                `"CUDA_VERSION_ARG=$CUDA_VERSION"

echo "--- Build libxgboost from the source"
$command_wrapper tests/ci_build/prune_libnccl.sh
$command_wrapper tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON -DUSE_NCCL=ON \
  -DUSE_OPENMP=ON -DHIDE_CXX_SYMBOLS=ON -DUSE_NCCL_LIB_PATH=ON \
  -DNCCL_INCLUDE_DIR=/usr/include -DNCCL_LIBRARY=/workspace/libnccl_static.a \
  ${arch_flag}
echo "--- Build binary wheel"
$command_wrapper bash -c \
  "cd python-package && rm -rf dist/* && python setup.py bdist_wheel --universal"
$command_wrapper python tests/ci_build/rename_whl.py python-package/dist/*.whl \
  ${BUILDKITE_COMMIT} ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with manylinux2014 standard"
tests/ci_build/ci_build.sh auditwheel_x86_64 docker auditwheel repair \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
$command_wrapper python tests/ci_build/rename_whl.py wheelhouse/*.whl \
  ${BUILDKITE_COMMIT} ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/
# Make sure that libgomp.so is vendored in the wheel
tests/ci_build/ci_build.sh auditwheel_x86_64 docker bash -c \
  "unzip -l python-package/dist/*.whl | grep libgomp  || exit -1"

echo "--- Upload Python wheel"
buildkite-agent artifact upload python-package/dist/*.whl
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  aws s3 cp python-package/dist/*.whl s3://xgboost-nightly-builds/${BRANCH_NAME}/ \
    --acl public-read --no-progress
fi
echo "-- Stash C++ test executable (testxgboost)"
buildkite-agent artifact upload build/testxgboost
