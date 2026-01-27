#!/bin/bash
## Build docs for the JVM packages and package it in a tarball.
## Note: this script assumes that the user has already built libxgboost4j.so
## and placed it in the lib/ directory.

set -euo pipefail

if [[ -z ${BRANCH_NAME:-} ]]; then
  echo "Make sure to define environment variable BRANCH_NAME."
  exit 1
fi

if [[ ! -f lib/libxgboost4j.so ]]; then
  echo "Must place libxgboost4j.so in lib/ first"
  exit 2
fi

echo "--- Build JVM packages doc"
set -x

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
tar cvjf ${BRANCH_NAME}.tar.bz2 javadocs/ scaladocs/
mv ${BRANCH_NAME}.tar.bz2 ..
cd ..
rm -rfv tmp/
