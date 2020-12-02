#!/usr/bin/env bash
set -e

if [[ "$1" == --conda-env=* ]]
then
  conda_env=$(echo "$1" | sed 's/^--conda-env=//g' -)
  echo "Activating Conda environment ${conda_env}"
  shift 1
  cmake_args="$@"
  source activate ${conda_env}
  cmake_prefix_flag="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
else
  cmake_args="$@"
  cmake_prefix_flag=''
fi

rm -rf build
mkdir build
cd build
cmake .. ${cmake_args} -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DCMAKE_VERBOSE_MAKEFILE=ON -DENABLE_ALL_WARNINGS=ON -GNinja ${cmake_prefix_flag}
ninja clean
time ninja -v
cd ..
