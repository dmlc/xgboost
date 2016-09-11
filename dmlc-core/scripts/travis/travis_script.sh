#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    make lint || exit -1
    make doxygen 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag" |grep warning) && exit -1
    exit 0
fi

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    export NO_OPENMP=1
fi

if [ ${TASK} == "unittest_gtest" ]; then
    cp make/config.mk .
    make -f scripts/packages.mk gtest
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        echo "USE_S3=1" >> config.mk
        echo "export CXX = g++-4.8" >> config.mk
    else
        echo "USE_S3=0" >> config.mk
        echo "USE_OPENMP=0" >> config.mk
    fi
    echo "GTEST_PATH="${CACHE_PREFIX} >> config.mk
    echo "BUILD_TEST=1" >> config.mk
    make all || exit -1
    test/unittest/dmlc_unittest || exit -1
fi
