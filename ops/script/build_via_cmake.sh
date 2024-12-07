#!/bin/bash

set -euo pipefail

if [[ "$#" -lt 1 ]]
then
  conda_env=""
else
  conda_env="$1"
fi

if [[ "${conda_env}" == --conda-env=* ]]
then
  conda_env=$(echo "${conda_env}" | sed 's/^--conda-env=//g' -)
  echo "Activating Conda environment ${conda_env}"
  shift 1
  cmake_args="$@"

  # Workaround for file permission error
  if [[ -n ${CI_BUILD_UID:-} ]]
  then
    gosu root chown -R "${CI_BUILD_UID}:${CI_BUILD_GID}" /opt/miniforge/envs
  fi

  # Don't activate Conda env if it's already activated
  if [[ -z ${CONDA_PREFIX:-} ]]
  then
    source activate ${conda_env}
  fi
  cmake_prefix_flag="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
else
  cmake_args="$@"
  cmake_prefix_flag=''
fi

rm -rf build
mkdir build
cd build
# Disable CMAKE_COMPILE_WARNING_AS_ERROR option temporarily until
# https://github.com/dmlc/xgboost/issues/10400 is fixed
set -x
cmake .. ${cmake_args} \
  -DGOOGLE_TEST=ON \
  -DUSE_DMLC_GTEST=ON \
  -DENABLE_ALL_WARNINGS=ON \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  -GNinja \
  ${cmake_prefix_flag} \
  -DHIDE_CXX_SYMBOLS=ON \
  -DBUILD_DEPRECATED_CLI=ON
ninja clean
time ninja -v
cd ..
set +x
