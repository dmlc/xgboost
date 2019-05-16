#!/bin/bash

cp make/travis.mk config.mk
make -f dmlc-core/scripts/packages.mk lz4

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    echo 'USE_OPENMP=0' >> config.mk
    echo 'TMPVAR := $(XGB_PLUGINS)' >> config.mk
    echo 'XGB_PLUGINS = $(filter-out plugin/lz4/plugin.mk, $(TMPVAR))' >> config.mk
else
    # use g++-4.8 for linux
    export CXX=g++-4.8
fi

if [ ${TASK} == "python_test" ]; then
    make all || exit -1
    echo "-------------------------------"
    source activate python3
    python --version
    conda install numpy scipy pandas matplotlib scikit-learn curl

    python -m pip install graphviz pytest pytest-cov codecov
    python -m pip install https://h2o-release.s3.amazonaws.com/datatable/stable/datatable-0.7.0/datatable-0.7.0-cp37-cp37m-linux_x86_64.whl
    python -m pytest -v --fulltrace -s tests/python --cov=python-package/xgboost || exit -1
    codecov
fi

if [ ${TASK} == "java_test" ]; then
    set -e
    cd jvm-packages
    mvn -q clean install -DskipTests -Dmaven.test.skip
    mvn -q test
fi

if [ ${TASK} == "cmake_test" ]; then
    set -e
    # Build gtest via cmake
    wget -nc https://github.com/google/googletest/archive/release-1.7.0.zip
    unzip -n release-1.7.0.zip
    mv googletest-release-1.7.0 gtest && cd gtest
    CC=gcc-7 CXX=g++-7 cmake . && make
    mkdir lib && mv libgtest.a lib
    cd ..
    rm -rf release-1.7.0.zip

    # Build/test
    rm -rf build
    mkdir build && cd build
    PLUGINS="-DPLUGIN_LZ4=ON -DPLUGIN_DENSE_PARSER=ON"
    CC=gcc-7 CXX=g++-7 cmake .. -DGOOGLE_TEST=ON -DGTEST_ROOT=$PWD/../gtest/ ${PLUGINS}
    make
    ./testxgboost
    cd ..
    rm -rf build
fi
