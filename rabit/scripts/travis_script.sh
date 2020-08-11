#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    make lint RABIT_BUILD_DMLC=1 || exit -1
fi

if [ ${TASK} == "doc" ]; then
    make doc 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag" |grep warning) && exit -1
fi

# we should depreciate Makefile based build
if [ ${TASK} == "build" ]; then
    make all RABIT_BUILD_DMLC=1 || exit -1
fi

if [ ${TASK} == "mpi-build" ]; then
    ./scripts/mpi_build.sh
    cd test
    make mpi RABIT_BUILD_DMLC=1 && make speed_test.mpi RABIT_BUILD_DMLC=1 || exit -1
fi
#
if [ ${TASK} == "cmake-test" ]; then
    mkdir build
    cd build
    cmake -DRABIT_BUILD_TESTS=ON -DRABIT_BUILD_DMLC=ON -DGTEST_ROOT=${HOME}/.local ..
    # known osx gtest 1.8 issue
    cp ${HOME}/.local/lib/*.dylib .
    make -j$(nproc)
    make test
    make install || exit -1
    cd ../test
    ../scripts/travis_runtest.sh || exit -1
    rm -rf ../build
fi
