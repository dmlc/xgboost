#!/bin/bash

if [ ${TASK} == "lint" ]; then
    make lint || exit -1
    echo "Check documentations..."
    make doxygen 2>log.txt
    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
    echo "---------Error Log----------"
    cat logclean.txt
    echo "----------------------------"
    (cat logclean.txt|grep warning) && exit -1
    (cat logclean.txt|grep error) && exit -1
    exit 0
fi

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
    conda install numpy scipy pandas matplotlib nose scikit-learn
    python -m pip install graphviz pytest pytest-cov codecov
    python -m nose tests/python || exit -1
    py.test tests/python --cov=python-package/xgboost
    codecov
    source activate python2
    echo "-------------------------------"
    python --version
    conda install numpy scipy pandas matplotlib nose scikit-learn
    python -m pip install graphviz
    python -m nose tests/python || exit -1
    exit 0
fi

if [ ${TASK} == "python_lightweight_test" ]; then
    make all || exit -1
    echo "-------------------------------"
    source activate python3
    python --version
    conda install numpy scipy nose
    python -m pip install graphviz pytest pytest-cov codecov
    python -m nose tests/python || exit -1
    py.test tests/python --cov=python-package/xgboost
    codecov
    source activate python2
    echo "-------------------------------"
    python --version
    conda install numpy scipy nose
    python -m pip install graphviz
    python -m nose tests/python || exit -1
    python -m pip install flake8
    flake8 --ignore E501 python-package || exit -1
    flake8 --ignore E501 tests/python || exit -1
    exit 0
fi

if [ ${TASK} == "r_test" ]; then
    set -e
    export _R_CHECK_TIMINGS_=0
    export R_BUILD_ARGS="--no-build-vignettes --no-manual"
    export R_CHECK_ARGS="--no-vignettes --no-manual"

    curl -OL http://raw.github.com/craigcitro/r-travis/master/scripts/travis-tool.sh
    chmod 755 ./travis-tool.sh
    ./travis-tool.sh bootstrap
    make Rpack
    cd ./xgboost
    ../travis-tool.sh install_deps
    ../travis-tool.sh run_tests
    exit 0
fi

if [ ${TASK} == "java_test" ]; then
    set -e
    cd jvm-packages
    mvn -q clean install -DskipTests -Dmaven.test.skip
    mvn -q test
fi

if [ ${TASK} == "cmake_test" ]; then
    mkdir build
    cd build
    cmake ..
    make
fi

if [ ${TASK} == "cpp_test" ]; then
    set -e
    make -f dmlc-core/scripts/packages.mk gtest
    echo "TEST_COVER=1" >> config.mk
    echo "GTEST_PATH="${CACHE_PREFIX} >> config.mk
    make cover
fi
