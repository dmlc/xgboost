#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 [branch name]"
  exit 1
fi

set -e
set -x

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

rm -rf build/
cd jvm-packages

branch_name=$1

# Install JVM packages in local Maven repository
mvn --no-transfer-progress install -DskipTests
# Build Scaladocs
mvn --no-transfer-progress scala:doc -DskipTests
# Build Javadocs
mvn --no-transfer-progress javadoc:javadoc -DskipTests

# Package JVM docs in a tarball
mkdir -p tmp/scaladocs
cp -rv xgboost4j/target/site/apidocs/ ./tmp/javadocs/
cp -rv xgboost4j/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j/
cp -rv xgboost4j-spark/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j-spark/
cp -rv xgboost4j-flink/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j-flink/

cd tmp
tar cvjf ${branch_name}.tar.bz2 javadocs/ scaladocs/
mv ${branch_name}.tar.bz2 ..
cd ..
rm -rfv tmp/

set +x
set +e
