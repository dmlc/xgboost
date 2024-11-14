#!/bin/bash

set -e
set -x

if [ $# -ne 1 ]; then
  echo "Usage: $0 [spark version]"
  exit 1
fi

spark_version=$1

cd jvm-packages
rm -rf $(find . -name target)
rm -rf ../build/

## Deploy JVM packages to xgboost-maven-repo

# Scala 2.12, CPU variant
mvn --no-transfer-progress deploy -Pdefault,release-to-s3 -Dspark.version=${spark_version} -DskipTests -Dmaven.test.skip=true
mvn clean
mvn clean -Pdefault,release-to-s3

# Scala 2.12, GPU variant
mvn --no-transfer-progress install -Pgpu -Dspark.version=${spark_version} -DskipTests -Dmaven.test.skip=true
mvn --no-transfer-progress deploy -Pgpu,release-to-s3 -pl xgboost4j-spark-gpu -Dspark.version=${spark_version} -DskipTests -Dmaven.test.skip=true

# Scala 2.13, CPU variant
pushd ..
python dev/change_scala_version.py --scala-version 2.13 --purge-artifacts
popd
mvn --no-transfer-progress deploy -Pdefault,release-to-s3 -Dspark.version=${spark_version} -DskipTests -Dmaven.test.skip=true
mvn clean
mvn clean -Pdefault,release-to-s3

# Scala 2.13, GPU variant
mvn --no-transfer-progress install -Pgpu -Dspark.version=${spark_version} -DskipTests -Dmaven.test.skip=true
mvn --no-transfer-progress deploy -Pgpu,release-to-s3 -pl xgboost4j-spark-gpu -Dspark.version=${spark_version} -DskipTests -Dmaven.test.skip=true

set +x
set +e
