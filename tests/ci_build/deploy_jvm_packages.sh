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

rm -rf build/
cd jvm-packages

# Re-build package without Mock Rabit
# Deploy to S3 bucket xgboost-maven-repo
mvn --no-transfer-progress package deploy -P release-to-s3 -Dspark.version=${spark_version} -DskipTests

set +x
set +e
