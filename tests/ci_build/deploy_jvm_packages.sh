#!/bin/bash

set -e
set -x

if [ $# -ne 2 ]; then
  echo "Usage: $0 [spark version] [build_gpu? 0 or 1]"
  exit 1
fi

spark_version=$1
build_gpu=$2

# Initialize local Maven repository
./tests/ci_build/initialize_maven.sh

cd jvm-packages
rm -rf $(find . -name target)
rm -rf ../build/

# Re-build package without Mock Rabit
# Deploy to S3 bucket xgboost-maven-repo
if [[ "$build_gpu" == "1" ]]
then
  # Build CPU artifact
  mvn --no-transfer-progress package deploy -P release-to-s3 -Dspark.version=${spark_version} -DskipTests
else
  # Build GPU artifact
  sed -i -e 's/<artifactId>xgboost\(.*\)_\(.*\)<\/artifactId>/<artifactId>xgboost\1-gpu_\2<\/artifactId>/' $(find . -name pom.xml)
  mvn --no-transfer-progress package deploy -Duse.cuda=ON -P release-to-s3 -Dspark.version=${spark_version} -DskipTests
fi

set +x
set +e
