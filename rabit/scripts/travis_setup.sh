#!/bin/bash

echo "Testing on: ${TRAVIS_OS_NAME}, Home directory: ${HOME}"

# Install Miniconda
if [ ${TRAVIS_OS_NAME} == "osx" ]; then
  wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
  wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
bash conda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -n python3 python=3.7
conda activate python3
conda --version
python --version
# Install Python packages
conda install -c conda-forge numpy scipy urllib3 websocket-client
python -m pip install cpplint pylint kubernetes

# Install googletest under home directory
GTEST_VERSION=1.8.1
GTEST_RELEASE=release-${GTEST_VERSION}.tar.gz
GTEST_TAR_BALL=googletest_${GTEST_RELEASE}

wget https://github.com/google/googletest/archive/${GTEST_RELEASE} -O ${GTEST_TAR_BALL}
echo "152b849610d91a9dfa1401293f43230c2e0c33f8 ${GTEST_TAR_BALL}" | sha1sum -c
tar -xf ${GTEST_TAR_BALL}
pushd .

cd googletest-release-${GTEST_VERSION}
mkdir build
cd build
echo "Installing to ${HOME}/.local"
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=${HOME}/.local
make -j$(nproc)
make install

popd

if [ ${TRAVIS_OS_NAME} == "linux" ]; then
    sudo apt-get install tree
fi

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew install python3
fi
