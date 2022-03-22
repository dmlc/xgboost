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
# Deploy to S3 bucket xgboost-maven-repo
mvn --no-transfer-progress package deploy -Duse.cuda=ON -P release-to-s3 -Dspark.version=${spark_version} -DskipTests

set +x
set +e
