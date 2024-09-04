#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ]
then
  mode=stable
  exit 1
else
  mode=$1
fi

WHEEL_TAG=manylinux_2_28_x86_64

source tests/buildkite/conftest.sh


case "${mode}" in
  stable)
    container_tag='gpu_build_rockylinux8'
    rapids_version=$RAPIDS_VERSION
    ;;

  dev)
    container_tag='gpu_dev_ver'
    rapids_version=$DEV_RAPIDS_VERSION
    ;;

  *)
    echo "Unrecognized mode ID: ${mode}"
    exit 2
    ;;
esac

echo "--- Build with CUDA ${CUDA_VERSION} with RMM"

if [[ ($is_pull_request == 1) || ($is_release_branch == 0) ]]
then
  arch_flag="-DGPU_COMPUTE_VER=75"
else
  arch_flag=""
fi

command_wrapper="tests/ci_build/ci_build.sh $container_tag --build-arg "`
                `"CUDA_VERSION_ARG=$CUDA_VERSION --build-arg "`
                `"NCCL_VERSION_ARG=$NCCL_VERSION --build-arg "`
                `"RAPIDS_VERSION_ARG=$rapids_version"

echo "--- Build libxgboost from the source"
$command_wrapper tests/ci_build/build_via_cmake.sh \
		 -DCMAKE_PREFIX_PATH="/opt/grpc;/opt/rmm;/opt/rmm/lib64/rapids/cmake" \
		 -DUSE_CUDA=ON \
		 -DUSE_OPENMP=ON \
		 -DHIDE_CXX_SYMBOLS=ON \
		 -DPLUGIN_FEDERATED=ON \
		 -DPLUGIN_RMM=ON \
		 -DUSE_NCCL=ON \
		 -DUSE_NCCL_LIB_PATH=ON \
		 -DNCCL_INCLUDE_DIR=/usr/include \
		 -DUSE_DLOPEN_NCCL=ON \
  ${arch_flag}
echo "--- Build binary wheel"
$command_wrapper bash -c \
  "cd python-package && rm -rf dist/* && pip wheel --no-deps -v . --wheel-dir dist/"
$command_wrapper python tests/ci_build/rename_whl.py  \
  --wheel-path python-package/dist/*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
tests/ci_build/ci_build.sh manylinux_2_28_x86_64 auditwheel repair \
  --plat ${WHEEL_TAG} python-package/dist/*.whl
$command_wrapper python tests/ci_build/rename_whl.py  \
  --wheel-path wheelhouse/*.whl  \
  --commit-hash ${BUILDKITE_COMMIT}  \
  --platform-tag ${WHEEL_TAG}
mv -v wheelhouse/*.whl python-package/dist/
# Make sure that libgomp.so is vendored in the wheel
tests/ci_build/ci_build.sh manylinux_2_28_x86_64 bash -c \
  "unzip -l python-package/dist/*.whl | grep libgomp  || exit -1"

echo "--- Upload Python wheel"
buildkite-agent artifact upload python-package/dist/*.whl
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  aws s3 cp python-package/dist/*.whl s3://xgboost-nightly-builds/experimental_build_with_rmm/ \
    --acl public-read --no-progress
fi

echo "-- Stash C++ test executable (testxgboost)"
buildkite-agent artifact upload build/testxgboost
