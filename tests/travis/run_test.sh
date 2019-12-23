#!/bin/bash

make -f dmlc-core/scripts/packages.mk lz4

if [ ${TASK} == "python_test" ]; then
    set -e
    # Build/test
    rm -rf build
    mkdir build && cd build
    cmake .. -DUSE_OPENMP=ON
    make -j2
    cd ..

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
    cmake .. -DGOOGLE_TEST=ON -DUSE_OPENMP=ON -DUSE_DMLC_GTEST=ON ${PLUGINS}
    make -j2
    ./testxgboost
    cd ..
    rm -rf build
fi
