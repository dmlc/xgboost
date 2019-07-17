#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    make lint RABIT_BUILD_DMLC=1 || exit -1
fi

if [ ${TASK} == "doc" ]; then
    make doc 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag" |grep warning) && exit -1
fi

if [ ${TASK} == "build" ]; then
    make all RABIT_BUILD_DMLC=1 || exit -1
fi

if [ ${TASK} == "mpi-build" ]; then
    ./scripts/mpi_build.sh
    cd test
    make mpi RABIT_BUILD_DMLC=1 && make speed_test.mpi RABIT_BUILD_DMLC=1 || exit -1
fi

if [ ${TASK} == "test" ]; then
    cd test
    make all RABIT_BUILD_DMLC=1 || exit -1
    ../scripts/travis_runtest.sh || exit -1
fi

if [ ${TASK} == "cmake-build" ]; then
    mkdir build
    cd build
    cmake .. -DRABIT_BUILD_DMLC=ON
    make all || exit -1
fi