#!/bin/bash
# remove all #pragma's that suppress compiler warnings
set -e
set -x
for file in xgboost/src/dmlc-core/include/dmlc/*.h
do
  sed -i.bak -e 's/^.*#pragma GCC diagnostic.*$//' -e 's/^.*#pragma clang diagnostic.*$//' -e 's/^.*#pragma warning.*$//' "${file}"
done
for file in xgboost/src/dmlc-core/include/dmlc/*.h.bak
do
  rm "${file}"
done
set +x
set +e
