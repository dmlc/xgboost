#!/bin/bash
# Inject lib/libxgboost4j.so into JVM packages.
# This script is useful when the user opts to set skip.native.build=true
# option in the JVM package build. When this option is set, the JVM package
# build will not build libxgboost4j.so; instead it will expect to find the
# library in jvm-packages/xgboost4j/src/main/resources/lib/{os}/{arch}/.
# This script will ensure that libxgboost4j.so is copied to the correct
# location.

set -euox pipefail

echo "Using externally provided libxgboost4j.so. Locating one from lib/..."
mkdir -p jvm-packages/xgboost4j/src/main/resources/lib/linux/x86_64/
cp -v lib/libxgboost4j.so jvm-packages/xgboost4j/src/main/resources/lib/linux/x86_64/
mkdir -p jvm-packages/xgboost4j/src/test/resources
mkdir -p jvm-packages/xgboost4j-spark/src/test/resources
mkdir -p jvm-packages/xgboost4j-spark-gpu/src/test/resources

# Generate machine.txt.* files from the CLI regression demo
# TODO(hcho3): Remove once CLI is removed
pushd demo/CLI/regression
python3 mapfeat.py
python3 mknfold.py machine.txt 1
popd

cp -v demo/data/agaricus.* \
  jvm-packages/xgboost4j/src/test/resources
cp -v demo/CLI/regression/machine.txt.t* demo/data/agaricus.* \
  jvm-packages/xgboost4j-spark/src/test/resources
cp -v demo/data/veterans_lung_cancer.csv \
  jvm-packages/xgboost4j-spark/src/test/resources/rank.train.csv \
  jvm-packages/xgboost4j-spark-gpu/src/test/resources
