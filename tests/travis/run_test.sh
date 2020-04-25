#!/bin/bash

make -f dmlc-core/scripts/packages.mk lz4

source $HOME/miniconda/bin/activate

if [ ${TASK} == "python_sdist_test" ]; then
    set -e

    conda activate python3
    python --version
    conda install numpy scipy

    make pippack
    python -m pip install xgboost-*.tar.gz -v --user
    python -c 'import xgboost' || exit -1
fi

if [ ${TASK} == "python_test" ]; then
    set -e
    # Build/test
    rm -rf build
    mkdir build && cd build
    cmake .. -DUSE_OPENMP=ON -DCMAKE_VERBOSE_MAKEFILE=ON
    make -j$(nproc)

    echo "-------------------------------"
    conda activate python3
    conda --version
    python --version

    # Build binary wheel
    cd ../python-package
    python setup.py bdist_wheel
    TAG=macosx_10_13_x86_64.macosx_10_14_x86_64.macosx_10_15_x86_64
    python ../tests/ci_build/rename_whl.py dist/*.whl ${TRAVIS_COMMIT} ${TAG}
    python -m pip install ./dist/xgboost-*-py3-none-${TAG}.whl

    # Run unit tests
    cd ..
    python -m pip install graphviz pytest pytest-cov codecov
    python -m pip install datatable
    python -m pip install numpy scipy pandas matplotlib scikit-learn dask[complete]
    python -m pytest -v --fulltrace -s tests/python --cov=python-package/xgboost || exit -1
    codecov

    # Deploy binary wheel to S3
    python -m pip install awscli
    if [ "${TRAVIS_PULL_REQUEST}" != "false" ]
    then
        S3_DEST="s3://xgboost-nightly-builds/PR-${TRAVIS_PULL_REQUEST}/"
    else
        if [ "${TRAVIS_BRANCH}" == "master" ]
        then
            S3_DEST="s3://xgboost-nightly-builds/"
        elif [ -z "${TRAVIS_TAG}" ]
        then
            S3_DEST="s3://xgboost-nightly-builds/${TRAVIS_BRANCH}/"
        fi
    fi
    python -m awscli s3 cp python-package/dist/*.whl "${S3_DEST}" --acl public-read || true
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
    cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON -DGOOGLE_TEST=ON -DUSE_OPENMP=ON -DUSE_DMLC_GTEST=ON ${PLUGINS}
    make -j$(nproc)
    ./testxgboost
    cd ..
    rm -rf build
fi
