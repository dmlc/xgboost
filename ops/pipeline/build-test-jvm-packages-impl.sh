#!/bin/bash
## Build and test JVM packages.
##
## Note. This script takes in all inputs via environment variables.

INPUT_DOC=$(
cat <<-EOF
Inputs
  - SCALA_VERSION:     Scala version, either 2.12 or 2.13 (Required)
  - USE_CUDA:          Set to 1 to enable CUDA
  - CUDA_ARCH:         Semicolon separated list of GPU compute capability targets
                       (e.g. '35;61') Only applicable if USE_CUDA=1
  - SKIP_NATIVE_BUILD: Set to 1 to have the JVM packages use an externally provided
                       libxgboost4j.so. (Usually Maven will invoke create_jni.py to
                       build it from scratch.) When using this option, make sure to
                       place libxgboost4j.so in lib/ directory.
EOF
)

set -euo pipefail

for arg in "SCALA_VERSION"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n${INPUT_DOC}"
    exit 1
  fi
done

set -x

# Set Scala version
if [[ "${SCALA_VERSION}" == "2.12" || "${SCALA_VERSION}" == "2.13" ]]
then
  python ops/change_scala_version.py --scala-version ${SCALA_VERSION} --purge-artifacts
else
  echo "Error: SCALA_VERSION must be either 2.12 or 2.13"
  exit 2
fi

# If SKIP_NATIVE_BUILD is set, copy in libxgboost4j.so from lib/
if [[ "${SKIP_NATIVE_BUILD:-}" == "1" ]]
then
  echo "Using externally provided libxgboost4j.so. Locating one from lib/..."
  cp -v lib/libxgboost4j.so ./jvm-packages/xgboost4j/src/main/resources/lib/linux/x86_64/
fi

cd jvm-packages/

# Ensure that XGBoost4J-Spark is compatible with multiple versions of Spark
if [[ "${USE_CUDA:-}" != "1" && "${SCALA_VERSION}" == "2.12" ]]
then
  for spark_version in 3.1.3 3.2.4 3.3.4 3.4.3
  do
    mvn --no-transfer-progress clean package -Dspark.version=${spark_version} \
      -pl xgboost4j,xgboost4j-spark
  done
fi

set +x
mvn_options=""
if [[ "${USE_CUDA:-}" == "1" ]]
then
  mvn_options="${mvn_options} -Pgpu"
fi
if [[ "${SKIP_NATIVE_BUILD:-}" == "1" ]]
then
  mvn_options="${mvn_options} -Dskip.native.build=true"
fi
set -x

if [[ -n "${CUDA_ARCH:-}" ]]
then
  export GPU_ARCH_FLAG="-DGPU_COMPUTE_VER='${CUDA_ARCH}'"
fi

mvn --no-transfer-progress clean install ${mvn_options}

# Integration tests
if [[ "${USE_CUDA:-}" != "1" ]]
then
  mvn --no-transfer-progress test -pl xgboost4j-example
fi
