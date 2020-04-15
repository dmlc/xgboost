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
# re-build package without Mock Rabit
mvn --no-transfer-progress package -Dspark.version=${spark_version} -DskipTests

# deploy to S3 bucket xgboost-maven-repo
mvn --no-transfer-progress deploy -P release-to-s3 -DskipTests

set +x
set +e
