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
cmake .. -DBUILD_C_DOC=ON
make -j

tar cvjf ${branch_name}.tar.bz2 doc_doxygen/
