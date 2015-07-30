#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        make lint  || exit -1
    fi
fi

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    export no_omp=1
    export NO_OPENMP=1
fi

if [ ${TASK} == "build" ]; then
    make all CXX=${CXX} || exit -1
fi

if [ ${TASK} == "build-with-dmlc" ]; then
    cd dmlc-core
    cp make/config.mk .
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        echo "USE_S3=1" >> config.mk
    else
        echo "USE_S3=0" >> config.mk
    fi
    make all CXX=${CXX}|| exit -1
    cd ..
    make dmlc=dmlc-core CXX=${CXX} || exit -1
fi

if [ ${TASK} == "R-package" ]; then
    scripts/travis_R_script.sh || exit -1
fi

if [ ${TASK} == "python-package" ]; then
    make all CXX=${CXX} || exit -1
    nosetests tests/python || exit -1
fi

# only test java under linux for now
if [ ${TASK} == "java-package" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        make java CXX=${CXX} || exit -1
        scripts/travis_java_script.sh || exit -1
    fi
fi
