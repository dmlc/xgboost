#!/bin/bash
## Build docs for the JVM packages and package it in a tarball.
## The docs Maven profile skips native compilation and tests, so no JNI library is needed.

set -euo pipefail

if [[ -z ${BRANCH_NAME:-} ]]; then
  echo "Make sure to define environment variable BRANCH_NAME."
  exit 1
fi

echo "--- Build JVM packages doc"
set -x

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
