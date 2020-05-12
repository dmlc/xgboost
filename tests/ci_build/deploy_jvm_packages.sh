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

# Compile XGBoost4J with Scala 2.11 too
mvn clean
# Rename artifactId of all XGBoost4J packages with suffix _2.11
sed -i -e 's/<artifactId>xgboost\(.*\)_[0-9\.]\+/<artifactId>xgboost\1_2.11/' $(find . -name pom.xml)
# Modify scala.version and scala.binary.version fields
sed -i -e 's/<scala\.version>[0-9\.]\+/<scala.version>2.11.12/' $(find . -name pom.xml)
sed -i -e 's/<scala\.binary\.version>[0-9\.]\+/<scala.binary.version>2.11/' $(find . -name pom.xml)

# Re-build and deploy
mvn --no-transfer-progress package deploy -P release-to-s3 -Dspark.version=${spark_version} -DskipTests

set +x
set +e
