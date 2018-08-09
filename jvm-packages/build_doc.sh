#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 [commit id]"
  exit 1
fi

set -e
set -x

commit_id=$1

# Install JVM packages in local Maven repository
mvn install -DskipTests
# Build Scaladocs
mvn scala:doc -DskipTests
# Build Javadocs
mvn javadoc:javadoc -DskipTests

# Package JVM docs in a tarball
mkdir -p tmp/scaladocs
cp -rv xgboost4j/target/site/apidocs/ ./tmp/javadocs/
cp -rv xgboost4j/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j/
cp -rv xgboost4j-spark/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j-spark/
cp -rv xgboost4j-flink/target/site/scaladocs/ ./tmp/scaladocs/xgboost4j-flink/

cd tmp
tar cvjf ${commit_id}.tar.bz2 javadocs/ scaladocs/
mv ${commit_id}.tar.bz2 ..
cd ..
rm -rfv tmp/

set +x
set +e
