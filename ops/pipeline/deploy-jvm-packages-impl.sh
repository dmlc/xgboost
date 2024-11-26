#!/bin/bash
## Deploy JVM packages to S3 bucket
## Companion script for ops/pipeline/deploy-jvm-packages.sh

set -euox pipefail

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {cpu,gpu} [scala_version]"
  exit 1
fi

variant="$1"
scala_version="$2"
maven_options="-DskipTests -Dmaven.test.skip=true -Dskip.native.build=true"

case "$variant" in
  cpu)
    # CPU variant
    python ops/script/change_scala_version.py --scala-version ${scala_version} --purge-artifacts
    bash ops/script/inject_jvm_lib.sh
    pushd jvm-packages
    mvn --no-transfer-progress deploy -Pdefault,release-to-s3 ${maven_options}
    popd
    ;;
  gpu)
    # GPU variant
    python ops/script/change_scala_version.py --scala-version ${scala_version} --purge-artifacts
    bash ops/script/inject_jvm_lib.sh
    pushd jvm-packages
    mvn --no-transfer-progress install -Pgpu ${maven_options}
    mvn --no-transfer-progress deploy -Pgpu,release-to-s3 -pl xgboost4j-spark-gpu ${maven_options}
    popd
    ;;
  *)
    echo "Unrecognized argument: $variant"
    exit 2
    ;;
esac
