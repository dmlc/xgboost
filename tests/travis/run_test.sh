#!/bin/bash

if [ ${TASK} == "lint" ]; then
    make lint || exit -1
    echo "Check documentations..."

    mkdir build_doc
    cd build_doc
    cmake .. -DBUILD_C_DOC=ON
    make doc_doxygen 2> log.txt

    (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
    echo "---------Error Log----------"
    cat logclean.txt
    echo "----------------------------"
    (cat logclean.txt|grep warning) && exit -1
    (cat logclean.txt|grep error) && exit -1

    cd -
    rm -rf build_doc

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
    conda install numpy scipy pandas matplotlib scikit-learn

    # Install data table from source
    wget http://releases.llvm.org/5.0.2/clang+llvm-5.0.2-x86_64-linux-gnu-ubuntu-14.04.tar.xz
    tar xf clang+llvm-5.0.2-x86_64-linux-gnu-ubuntu-14.04.tar.xz
    export LLVM5=$(pwd)/clang+llvm-5.0.2-x86_64-linux-gnu-ubuntu-14.04
    python -m pip install datatable --no-binary datatable

    python -m pip install graphviz pytest pytest-cov codecov
    python -m pytest -v --fulltrace -s tests/python --cov=python-package/xgboost || exit -1
    codecov

    source activate python2
    echo "-------------------------------"
    python --version
    conda install numpy scipy pandas matplotlib scikit-learn
    python -m pip install graphviz pytest
    python -m pytest -v --fulltrace -s tests/python || exit -1
    exit 0
fi

if [ ${TASK} == "python_lightweight_test" ]; then
    make all || exit -1

    echo "-------------------------------"
    source activate python3
    python --version
    conda install numpy scipy
    python -m pip install graphviz pytest pytest-cov codecov
    python -m pytest -v --fulltrace -s tests/python --cov=python-package/xgboost || exit -1
    codecov

    source activate python2
    echo "-------------------------------"
    python --version
    conda install numpy scipy pytest
    python -m pip install graphviz
    python -m pip install flake8==3.4.1
    python -m pytest -v --fulltrace -s tests/python || exit -1

    flake8 --ignore E501 python-package || exit -1
    flake8 --ignore E501 tests/python || exit -1
    exit 0
fi

if [ ${TASK} == "r_test" ]; then
    set -e

    make Rpack
    cd ./xgboost

    # Install package deps
    Rscript -e "install.packages( \
        c('devtools', 'testthat', 'lintr') \
        , repos = 'http://cloud.r-project.org' \
        , dependencies = c('Depends', 'Imports', 'LinkingTo') \
    )"

    Rscript -e \
        "devtools::install_deps( \
            repos = 'http://cloud.r-project.org' \
            , upgrade = 'never' \
            , dependencies = c('Depends', 'Imports', 'LinkingTo') \
        )"

    # install suggested packages separately to avoid huge build times
    Rscript -e "install.packages( \
        c('DiagrammeR', 'Ckmeans.1d.dp', 'vcd') \
        , repos = 'https://cloud.r-project.org' \
        , dependencies = c('Depends', 'Imports', 'LinkingTo') \
    )"

    # Run tests
    echo "Building with R CMD build"
    R CMD build \
        --no-build-vignettes \
        --no-manual \
        .

    echo "Running R tests"
    R_PACKAGE_TARBALL=$(ls -1t *.tar.gz | head -n 1)

    export _R_CHECK_TIMINGS_=0
    export _R_CHECK_FORCE_SUGGESTS_=false
    R CMD check \
        ${R_PACKAGE_TARBALL} \
        --no-vignettes \
        --no-manual \
        --as-cran \
        --install-args=--build

    exit 0
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

if [ ${TASK} == "cpp_test" ]; then
    set -e
    make -f dmlc-core/scripts/packages.mk gtest
    echo "TEST_COVER=1" >> config.mk
    echo "GTEST_PATH="${CACHE_PREFIX} >> config.mk
    make cover
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
