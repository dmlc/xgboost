#!/usr/bin/env bash
#
# Execute command within a docker container
#
# Usage: ci_build.sh <CONTAINER_TYPE> <DOCKER_BINARY>
#                    [--dockerfile <DOCKERFILE_PATH>] [-it]
#                    [--build-arg <BUILD_ARG>] <COMMAND>
#
# CONTAINER_TYPE: Type of the docker container used the run the build: e.g.,
#                 (cpu | gpu)
#
# DOCKER_BINARY: Command to invoke docker, e.g. (docker | nvidia-docker).
#
# DOCKERFILE_PATH: (Optional) Path to the Dockerfile used for docker build.  If
#                  this optional value is not supplied (via the --dockerfile
#                  flag), will use Dockerfile.CONTAINER_TYPE in default
#
# BUILD_ARG: (Optional) an argument to be passed to docker build
#
# COMMAND: Command to be executed in the docker container
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the command line arguments.
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Dockerfile to be used in docker build
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}"
DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"

# Get docker binary command (should be either docker or nvidia-docker)
DOCKER_BINARY="$1"
shift 1

if [[ "$1" == "--dockerfile" ]]; then
    DOCKERFILE_PATH="$2"
    DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
    echo "Using custom Dockerfile path: ${DOCKERFILE_PATH}"
    echo "Using custom docker build context path: ${DOCKER_CONTEXT_PATH}"
    shift 2
fi

if [[ -n "${CI_DOCKER_EXTRA_PARAMS_INIT}" ]]
then
    IFS=' ' read -r -a CI_DOCKER_EXTRA_PARAMS <<< "${CI_DOCKER_EXTRA_PARAMS_INIT}"
fi

if [[ "$1" == "-it" ]]; then
    CI_DOCKER_EXTRA_PARAMS+=('-it')
    shift 1
fi

while [[ "$1" == "--build-arg" ]]; do
    CI_DOCKER_BUILD_ARG+=" $1"
    CI_DOCKER_BUILD_ARG+=" $2"
    shift 2
done

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    echo "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
    exit 1
fi

COMMAND=("$@")

# Validate command line arguments.
if [ "$#" -lt 1 ] || [ ! -e "${SCRIPT_DIR}/Dockerfile.${CONTAINER_TYPE}" ]; then
    supported_container_types=$( ls -1 ${SCRIPT_DIR}/Dockerfile.* | \
        sed -n 's/.*Dockerfile\.\([^\/]*\)/\1/p' | tr '\n' ' ' )
      echo "Usage: $(basename $0) CONTAINER_TYPE COMMAND"
      echo "       CONTAINER_TYPE can be one of [${supported_container_types}]"
      echo "       COMMAND is a command (with arguments) to run inside"
      echo "               the container."
      exit 1
fi

# Helper function to traverse directories up until given file is found.
function upsearch () {
    test / == "$PWD" && return || \
        test -e "$1" && echo "$PWD" && return || \
        cd .. && upsearch "$1"
}

# Set up WORKSPACE. Jenkins will set them for you or we pick
# reasonable defaults if you run it outside of Jenkins.
WORKSPACE="${WORKSPACE:-${SCRIPT_DIR}/../../}"

# Determine the docker image name
DOCKER_IMG_NAME="xgb-ci.${CONTAINER_TYPE}"

# Append cuda version if available
CUDA_VERSION=$(echo "${CI_DOCKER_BUILD_ARG}" | egrep -o 'CUDA_VERSION=[0-9]+\.[0-9]+' | egrep -o '[0-9]+\.[0-9]+')
# Append jdk version if available
JDK_VERSION=$(echo "${CI_DOCKER_BUILD_ARG}" | egrep -o 'JDK_VERSION=[0-9]+' | egrep -o '[0-9]+')
# Append cmake version if available
CMAKE_VERSION=$(echo "${CI_DOCKER_BUILD_ARG}" | egrep -o 'CMAKE_VERSION=[0-9]+\.[0-9]+' | egrep -o '[0-9]+\.[0-9]+')
# Append R version if available
USE_R35=$(echo "${CI_DOCKER_BUILD_ARG}" | egrep -o 'USE_R35=[0-9]+' | egrep -o '[0-9]+$')
if [[ ${USE_R35} == "1" ]]; then
  USE_R35="_r35"
elif [[ ${USE_R35} == "0" ]]; then
  USE_R35="_no_r35"
fi
DOCKER_IMG_NAME=$DOCKER_IMG_NAME$CUDA_VERSION$JDK_VERSION$CMAKE_VERSION$USE_R35

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Bash on Ubuntu on Windows
UBUNTU_ON_WINDOWS=$([ -e /proc/version ] && grep -l Microsoft /proc/version || echo "")
# MSYS, Git Bash, etc.
MSYS=$([ -e /proc/version ] && grep -l MINGW /proc/version || echo "")

if [[ -z "$UBUNTU_ON_WINDOWS" ]] && [[ -z "$MSYS" ]] && [[ ! "$OSTYPE" == "darwin"* ]]; then
    USER_IDS="-e CI_BUILD_UID=$( id -u ) -e CI_BUILD_GID=$( id -g ) -e CI_BUILD_USER=$( id -un ) -e CI_BUILD_GROUP=$( id -gn ) -e CI_BUILD_HOME=${WORKSPACE}"
fi

# Print arguments.
cat <<EOF
   WORKSPACE: ${WORKSPACE}
   CI_DOCKER_EXTRA_PARAMS: ${CI_DOCKER_EXTRA_PARAMS[*]}
   CI_DOCKER_BUILD_ARG: ${CI_DOCKER_BUILD_ARG}
   COMMAND: ${COMMAND[*]}
   CONTAINER_TYPE: ${CONTAINER_TYPE}
   BUILD_TAG: ${BUILD_TAG}
   NODE_NAME: ${NODE_NAME}
   DOCKER CONTAINER NAME: ${DOCKER_IMG_NAME}
   USER_IDS: ${USER_IDS}
EOF


# Build the docker container.
echo "Building container (${DOCKER_IMG_NAME})..."

# If enviornment variables DOCKER_CACHE_ECR_ID and DOCKER_CACHE_ECR_REGION are set, use AWS ECR for build caching
if [[ -n "${DOCKER_CACHE_ECR_ID}" && -n "${DOCKER_CACHE_ECR_REGION}" ]]
then
    # Format Docker repo URL
    DOCKER_CACHE_REPO="${DOCKER_CACHE_ECR_ID}.dkr.ecr.${DOCKER_CACHE_ECR_REGION}.amazonaws.com"
    echo "Using AWS ECR; repo URL = ${DOCKER_CACHE_REPO}"
    # Login for Docker registry
    echo "\$(python3 -m awscli ecr get-login --no-include-email --region ${DOCKER_CACHE_ECR_REGION} --registry-ids ${DOCKER_CACHE_ECR_ID})"
    $(python3 -m awscli ecr get-login --no-include-email --region ${DOCKER_CACHE_ECR_REGION} --registry-ids ${DOCKER_CACHE_ECR_ID})
    # Pull pre-build container from Docker build cache,
    # if one exists for the particular branch or pull request
    echo "docker pull ${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
    if docker pull "${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
    then
      CACHE_FROM_CMD="--cache-from ${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
    else
      # If the build cache is empty of the particular branch or pull request,
      # use the build cache associated with the master branch
      echo "docker pull ${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:master"
      docker pull "${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:master" || true
      CACHE_FROM_CMD="--cache-from ${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:master"
    fi
else
    CACHE_FROM_CMD=''
fi

echo "docker build \
    ${CI_DOCKER_BUILD_ARG} \
    -t ${DOCKER_IMG_NAME} \
    -f ${DOCKERFILE_PATH} ${DOCKER_CONTEXT_PATH} \
    ${CACHE_FROM_CMD}"
docker build \
    ${CI_DOCKER_BUILD_ARG} \
    -t "${DOCKER_IMG_NAME}" \
    -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}" \
    ${CACHE_FROM_CMD}

# Check docker build status
if [[ $? != "0" ]]; then
    echo "ERROR: docker build failed."
    exit 1
fi

# If enviornment variable DOCKER_CACHE_REPO is set, use an external Docker repo for build caching
if [[ -n "${DOCKER_CACHE_REPO}" ]]
then
    # Push the container we just built to the Docker build cache
    # that is associated with the particular branch or pull request
    echo "docker tag ${DOCKER_IMG_NAME} ${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
    docker tag "${DOCKER_IMG_NAME}" "${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
    echo "docker push ${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
    docker push "${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
    if [[ $? != "0" ]]; then
        echo "ERROR: could not update Docker cache ${DOCKER_CACHE_REPO}/${DOCKER_IMG_NAME}:${BRANCH_NAME}"
        exit 1
    fi
fi


# Run the command inside the container.
echo "Running '${COMMAND[*]}' inside ${DOCKER_IMG_NAME}..."

# By default we cleanup - remove the container once it finish running (--rm)
# and share the PID namespace (--pid=host) so the process inside does not have
# pid 1 and SIGKILL is propagated to the process inside (jenkins can kill it).
set -x
${DOCKER_BINARY} run --rm --pid=host \
    -v "${WORKSPACE}":/workspace \
    -w /workspace \
    ${USER_IDS} \
    "${CI_DOCKER_EXTRA_PARAMS[@]}" \
    "${DOCKER_IMG_NAME}" \
    "${COMMAND[@]}"

