#!/bin/bash
## Deploy JVM packages to xgboost-maven-repo S3 bucket

set -euox pipefail

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {cpu,gpu}"
  exit 1
fi

variant="$1"

maven_options="-DskipTests -Dmaven.test.skip=true -Dskip.native.build=true"
case "$variant" in
  cpu)
    # CPU variant
    for scala_version in 2.12 2.13
    do
      python ops/script/change_scala_version.py --scala-version ${scala_version} --purge-artifacts
      pushd jvm-packages
      mvn --no-transfer-progress deploy -Pdefault,release-to-s3 ${maven_options}
      mvn clean
      mvn clean -Pdefault,release-to-s3
      popd
    done
    ;;
  gpu)
    # GPU variant
    for scala_version in 2.12 2.13
    do
      python ops/script/change_scala_version.py --scala-version ${scala_version} --purge-artifacts
      pushd jvm-packages
      mvn --no-transfer-progress install -Pgpu ${maven_options}
      mvn --no-transfer-progress deploy -Pgpu,release-to-s3 -pl xgboost4j-spark-gpu ${maven_options}
      mvn clean
      mvn clean -Pgpu,release-to-s3
      popd
    done
    ;;
  *)
    echo "Unrecognized argument: $variant"
    exit 2
    ;;
esac
