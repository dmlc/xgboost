#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 [branch name]"
  exit 1
fi

set -e
set -x

branch_name=$1

rm -rf build
mkdir build
cd build
cmake .. -GNinja -DBUILD_C_DOC=ON -DCMAKE_VERBOSE_MAKEFILE=ON
ninja -v

tar cvjf ${branch_name}.tar.bz2 doc_doxygen/
