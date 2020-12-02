#!/bin/bash

make -f dmlc-core/scripts/packages.mk lz4

source $HOME/miniconda/bin/activate

if [ ${TASK} == "python_sdist_test" ]; then
    set -e

    conda activate python3
    python --version
    cmake --version

    make pippack
    python -m pip install xgboost-*.tar.gz -v --user
    python -c 'import xgboost' || exit -1
fi

if [ ${TASK} == "python_test" ]; then
    if grep -n -R '<<<.*>>>\(.*\)' src include | grep --invert "NOLINT"; then
        echo 'Do not use raw CUDA execution configuration syntax with <<<blocks, threads>>>.' \
             'try `dh::LaunchKernel`'
        exit -1
    fi

    set -e


    # Build binary wheel
    if [ ${TRAVIS_CPU_ARCH} == "arm64" ]; then
      # Build manylinux2014 wheel on ARM64
      tests/ci_build/ci_build.sh aarch64 docker tests/ci_build/build_via_cmake.sh --conda-env=aarch64_test
      tests/ci_build/ci_build.sh aarch64 docker bash -c "cd build && ctest --extra-verbose"
      tests/ci_build/ci_build.sh aarch64 docker bash -c "cd python-package && rm -rf dist/* && python setup.py bdist_wheel --universal"
      TAG=manylinux2014_aarch64
      tests/ci_build/ci_build.sh aarch64 docker python tests/ci_build/rename_whl.py python-package/dist/*.whl ${TRAVIS_COMMIT} ${TAG}
    else
      rm -rf build
      mkdir build && cd build
      conda activate python3
      cmake --version
      cmake .. -DUSE_OPENMP=ON -DCMAKE_VERBOSE_MAKEFILE=ON
      make -j$(nproc)
      cd ../python-package
      python setup.py bdist_wheel
      cd ..
      TAG=macosx_10_14_x86_64.macosx_10_15_x86_64.macosx_11_0_x86_64
      python tests/ci_build/rename_whl.py python-package/dist/*.whl ${TRAVIS_COMMIT} ${TAG}
    fi

    # Run unit tests
    echo "------------------------------"
    if [ ${TRAVIS_CPU_ARCH} == "arm64" ]; then
        tests/ci_build/ci_build.sh aarch64 docker \
          bash -c "source activate aarch64_test && python -m pip install ./python-package/dist/xgboost-*-py3-none-${TAG}.whl && python -m pytest -v -s -rxXs --durations=0 --fulltrace tests/python/test_basic.py tests/python/test_basic_models.py tests/python/test_model_compatibility.py --cov=python-package/xgboost"
    else
        conda env create -n cpu_test --file=tests/ci_build/conda_env/macos_cpu_test.yml
        conda activate cpu_test
        python -m pip install ./python-package/dist/xgboost-*-py3-none-${TAG}.whl
        conda --version
        python --version
        python -m pytest -v -s -rxXs --durations=0 --fulltrace tests/python --cov=python-package/xgboost || exit -1
    fi
    conda activate python3
    codecov

    # Deploy binary wheel to S3
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

if [ ${TASK} == "s390x_test" ]; then
    set -e

    # Build and run C++ tests
    rm -rf build
    mkdir build && cd build
    cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON -DGOOGLE_TEST=ON -DUSE_OPENMP=ON -DUSE_DMLC_GTEST=ON -GNinja
    time ninja -v
    ./testxgboost

    # Run model compatibility tests
    cd ..
    python3 -m pip install --user pytest hypothesis
    PYTHONPATH=./python-package python3 -m pytest --fulltrace -v -rxXs tests/python/ -k 'test_model'
fi
