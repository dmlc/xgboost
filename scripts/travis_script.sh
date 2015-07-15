#!/bin/bash

# main script of travis
if [ ${TASK} == "lint" ]; then
    make lint  || exit -1
fi

if [ ${TASK} == "build" ]; then
    make all CXX=${CXX} || exit -1
fi

if [ ${TASK} == "build-with-dmlc" ]; then
    cd dmlc-core
    cp make/config.mk .
    echo "USE_S3=1" >> config.mk
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

if [ ${TASK} == "java-package" ]; then
    make java CXX=${CXX} || exit -1
    scripts/travis_java_script.sh || exit -1
fi
