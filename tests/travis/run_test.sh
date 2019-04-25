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
    conda install numpy scipy pandas matplotlib scikit-learn

    python -m pip install graphviz pytest pytest-cov codecov
    python -m pip install https://h2o-release.s3.amazonaws.com/datatable/stable/datatable-0.7.0/datatable-0.7.0-cp37-cp37m-linux_x86_64.whl
    python -m pytest -v --fulltrace -s tests/python --cov=python-package/xgboost || exit -1
    codecov

    flake8 --ignore E501 python-package || exit -1
    flake8 --ignore E501 tests/python || exit -1
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
    cmake . && make
    mkdir lib && mv libgtest.a lib
    cd ..
    rm -rf release-1.7.0.zip

    # Build/test
    rm -rf build
    mkdir build && cd build
    PLUGINS="-DPLUGIN_LZ4=ON -DPLUGIN_DENSE_PARSER=ON"
    cmake .. -DGOOGLE_TEST=ON -DGTEST_ROOT=$PWD/../gtest/ ${PLUGINS}
    make
    ./testxgboost
    cd ..
    rm -rf build
fi

if [ ${TASK} == "distributed_test" ]; then
    set -e
    make all || exit -1
    echo "-------------------------------"
    source activate python3
    python --version
    conda install numpy scipy
    python -m pip install kubernetes
    cd tests/distributed
    ./runtests.sh
fi

if [ ${TASK} == "sanitizer_test" ]; then
    set -e
    # Build gtest via cmake
    wget -nc https://github.com/google/googletest/archive/release-1.7.0.zip
    unzip -n release-1.7.0.zip
    mv googletest-release-1.7.0 gtest && cd gtest
    CC=gcc-7 CXX=g++-7 cmake -DCMAKE_CXX_FLAGS="-fuse-ld=gold" \
      -DCMAKE_C_FLAGS="-fuse-ld=gold"
    make
    mkdir lib && mv libgtest.a lib
    cd ..
    rm -rf release-1.7.0.zip

    mkdir build && cd build
    CC=gcc-7 CXX=g++-7 cmake .. -DGOOGLE_TEST=ON -DGTEST_ROOT=$PWD/../gtest/ \
      -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DSANITIZER_PATH=/usr/lib/x86_64-linux-gnu/ \
      -DCMAKE_CXX_FLAGS="-fuse-ld=gold" \
      -DCMAKE_C_FLAGS="-fuse-ld=gold"
    make

    export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
    ASAN_OPTIONS=symbolize=1 ./testxgboost
    cd ..
    rm -rf build
    exit 0
fi
