#!/bin/bash

export GTEST_PKG_NAME=release-1.8.1
export GTEST_DIR_NAME=googletest-${GTEST_PKG_NAME}  # uncompressed directory
export GTEST_ZIP_FILE=${GTEST_PKG_NAME}.zip	    # downloaded zip ball name

rm -rf gtest googletest-release*

wget -nc https://github.com/google/googletest/archive/${GTEST_ZIP_FILE}
unzip -n ${GTEST_ZIP_FILE}
mv ${GTEST_DIR_NAME} gtest && cd gtest
cmake . -DCMAKE_INSTALL_PREFIX=./ins && make
make install

cd ..
rm ${GTEST_ZIP_FILE}

python3 tests/ci_build/tidy.py --gtest-path=${PWD}/gtest/ins
