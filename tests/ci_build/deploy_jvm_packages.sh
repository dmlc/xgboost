#!/bin/bash

set -e
set -x

if [ $# -ne 1 ]; then
  echo "Usage: $0 [spark version]"
  exit 1
fi

spark_version=$1

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

cd jvm-packages
rm -rf $(find . -name target)
rm -rf ../build/

# Re-build package without Mock Rabit
# Maven profiles:
# `default`           includes modules: xgboost4j, xgboost4j-spark, xgboost4j-flink, xgboost4j-example
# `gpu`               includes modules: xgboost4j-gpu, xgboost4j-spark-gpu, sets `use.cuda = ON`
# `scala-2.13`        sets the scala binary version to the 2.13
# `release-to-s3`     sets maven deployment targets

# Deploy to S3 bucket xgboost-maven-repo
mvn --no-transfer-progress package deploy -P default,gpu,release-to-s3 -Dspark.version=${spark_version} -DskipTests
# Deploy scala 2.13 to S3 bucket xgboost-maven-repo
mvn --no-transfer-progress package deploy -P release-to-s3,default,scala-2.13 -Dspark.version=${spark_version} -DskipTests


set +x
set +e
