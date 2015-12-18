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

if [ ${TASK} == "python-package" -o ${TASK} == "python-package3" ]; then

    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        brew install graphviz
        if [ ${TASK} == "python-package3" ]; then
            wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        else
            wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
        fi
    else
        sudo apt-get install graphviz
        if [ ${TASK} == "python-package3" ]; then
            wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
        else
            wget -O conda.sh https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
        fi
    fi
    bash conda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    # Useful for debugging any issues with conda
    conda info -a

    if [ ${TASK} == "python-package3" ]; then
        conda create -n myenv python=3.4
    else
        conda create -n myenv python=2.7
    fi
    source activate myenv
    conda install numpy scipy pandas matplotlib nose scikit-learn
    python -m pip install graphviz

    make all CXX=${CXX} || exit -1

    python -m nose tests/python || exit -1
    python --version
fi

# only test java under linux for now
if [ ${TASK} == "java-package" ]; then
    if [ ${TRAVIS_OS_NAME} != "osx" ]; then
        make java CXX=${CXX} || exit -1
        scripts/travis_java_script.sh || exit -1
    fi
fi
