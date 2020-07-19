#!/usr/bin/env bash
set -e

if [[ "$1" == --conda-env=* ]]
then
  conda_env=$(echo "$1" | sed 's/^--conda-env=//g' -)
  echo "Activating Conda environment ${conda_env}"
  shift 1
  cmake_args="$@"
  source activate ${conda_env}
else
  cmake_args="$@"
fi

rm -rf build
mkdir build
cd build
cmake .. ${cmake_args} -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DCMAKE_VERBOSE_MAKEFILE=ON -GNinja
ninja clean
time ninja -v
cd ..
