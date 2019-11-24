#!/bin/bash

cp make/travis.mk config.mk
make -f dmlc-core/scripts/packages.mk lz4

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    echo 'USE_OPENMP=0' >> config.mk
fi

if [ ${TASK} == "python_test" ]; then
    make all || exit -1
    echo "-------------------------------"
    conda activate python3
    python --version
    conda install numpy scipy pandas matplotlib scikit-learn

    python -m pip install graphviz pytest pytest-cov codecov
    python -m pip install dask distributed dask[dataframe]
    python -m pip install datatable
    python -m pytest -v --fulltrace -s tests/python --cov=python-package/xgboost || exit -1
    codecov
fi

if [ ${TASK} == "java_test" ]; then
    export RABIT_MOCK=ON
    conda activate python3
    cd jvm-packages
    mvn -q clean install -DskipTests -Dmaven.test.skip
    mvn -q test
fi

if [ ${TASK} == "cmake_test" ]; then
    set -e

    if grep -n -R '<<<.*>>>\(.*\)' src include | grep --invert "NOLINT"; then
        echo 'Do not use raw CUDA execution configuration syntax with <<<blocks, threads>>>.' \
             'try `dh::LaunchKernel`'
        exit -1
    fi

    # Build/test
    rm -rf build
    mkdir build && cd build
    PLUGINS="-DPLUGIN_LZ4=ON -DPLUGIN_DENSE_PARSER=ON"
    CC=gcc-9 CXX=g++-9 cmake .. -DGOOGLE_TEST=ON -DUSE_OPENMP=ON -DUSE_DMLC_GTEST=ON ${PLUGINS}
    make
    ./testxgboost
    cd ..
    rm -rf build
fi
