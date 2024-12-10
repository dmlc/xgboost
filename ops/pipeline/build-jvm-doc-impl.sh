#!/bin/bash
## Build docs for the JVM packages and package it in a tarball
## Note: Note: this script assumes that the user has already built libxgboost4j.so
## and place it in the lib/ directory.

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 [branch name]"
  exit 1
fi

set -euo pipefail

branch_name=$1

# Copy in libxgboost4j.so
mkdir -p jvm-packages/xgboost4j/src/main/resources/lib/linux/x86_64/
cp -v lib/libxgboost4j.so jvm-packages/xgboost4j/src/main/resources/lib/linux/x86_64/

cd jvm-packages/
# Install JVM packages in local Maven repository
mvn --no-transfer-progress install -Pdocs
# Build Scaladocs
mvn --no-transfer-progress scala:doc -Pdocs
# Build Javadocs
mvn --no-transfer-progress javadoc:javadoc -Pdocs

# Package JVM docs in a tarball
mkdir -p tmp/scaladocs
cp -rv xgboost4j/target/reports/apidocs/ ./tmp/javadocs/
cp -rv xgboost4j/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j/
cp -rv xgboost4j-spark/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j-spark/
cp -rv xgboost4j-spark-gpu/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j-spark-gpu/
cp -rv xgboost4j-flink/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j-flink/

cd tmp
tar cvjf ${branch_name}.tar.bz2 javadocs/ scaladocs/
mv ${branch_name}.tar.bz2 ..
cd ..
rm -rfv tmp/

set +x
set +e
