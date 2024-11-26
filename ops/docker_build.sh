#!/bin/bash
## Build a CI container and cache the layers in AWS ECR (Elastic Container Registry).
## This script provides a convenient wrapper for ops/docker_build.py.
## Build-time variables (--build-arg) and container defintion are fetched from
## ops/docker/ci_container.yml.
##
## Note. This script takes in some inputs via environment variables.

USAGE_DOC=$(
cat <<-EOF
Usage: ops/docker_build.sh [container_id]

In addition, the following environment variables should be set.
  - BRANCH_NAME:      Name of the current git branch or pull request (Required)
  - USE_DOCKER_CACHE: If set to 1, enable caching
EOF
)

ECR_LIFECYCLE_RULE=$(
cat <<-EOF
{
   "rules": [
       {
           "rulePriority": 1,
           "selection": {
               "tagStatus": "any",
               "countType": "sinceImagePushed",
               "countUnit": "days",
               "countNumber": 30
           },
           "action": {
               "type": "expire"
           }
       }
   ]
}
EOF
)

set -euo pipefail

for arg in "BRANCH_NAME"
do
  if [[ -z "${!arg:-}" ]]
  then
    echo -e "Error: $arg must be set.\n\n${USAGE_DOC}"
    exit 1
  fi
done

if [[ "$#" -lt 1 ]]
then
  echo "${USAGE_DOC}"
  exit 2
fi
CONTAINER_ID="$1"

# Fetch CONTAINER_DEF and BUILD_ARGS
source <(ops/docker/extract_build_args.sh ${CONTAINER_ID} | tee /dev/stderr) 2>&1

if [[ "${USE_DOCKER_CACHE:-}" != "1" ]]   # Any value other than 1 is considered false
then
  USE_DOCKER_CACHE=0
fi

if [[ ${USE_DOCKER_CACHE} -eq 0 ]]
then
  echo "USE_DOCKER_CACHE not set; caching disabled"
else
  DOCKER_CACHE_ECR_ID=$(yq ".DOCKER_CACHE_ECR_ID" ops/docker/docker_cache_ecr.yml)
  DOCKER_CACHE_ECR_REGION=$(yq ".DOCKER_CACHE_ECR_REGION" ops/docker/docker_cache_ecr.yml)
  DOCKER_CACHE_REPO="${DOCKER_CACHE_ECR_ID}.dkr.ecr.${DOCKER_CACHE_ECR_REGION}.amazonaws.com"
  echo "Using AWS ECR; repo URL = ${DOCKER_CACHE_REPO}"
  # Login for Docker registry
  echo "aws ecr get-login-password --region ${DOCKER_CACHE_ECR_REGION} |" \
       "docker login --username AWS --password-stdin ${DOCKER_CACHE_REPO}"
  aws ecr get-login-password --region ${DOCKER_CACHE_ECR_REGION} \
    | docker login --username AWS --password-stdin ${DOCKER_CACHE_REPO}
fi

# Pull pre-built container from the cache
# First try locating one for the particular branch or pull request
CACHE_FROM_CMD=""
IS_CACHED=0
if [[ ${USE_DOCKER_CACHE} -eq 1 ]]
then
  DOCKER_TAG="${BRANCH_NAME//\//-}"  # Slashes are not allowed in Docker tag
  DOCKER_URL="${DOCKER_CACHE_REPO}/${CONTAINER_ID}:${DOCKER_TAG}"
  echo "docker pull --quiet ${DOCKER_URL}"
  if time docker pull --quiet "${DOCKER_URL}"
  then
    echo "Found a cached container for the branch ${BRANCH_NAME}: ${DOCKER_URL}"
    IS_CACHED=1
  else
    # If there's no pre-built container from the cache,
    # use the pre-built container from the master branch.
    DOCKER_URL="${DOCKER_CACHE_REPO}/${CONTAINER_ID}:master"
    echo "Could not find a cached container for the branch ${BRANCH_NAME}." \
         "Using a cached container from the master branch: ${DOCKER_URL}"
    echo "docker pull --quiet ${DOCKER_URL}"
    if time docker pull --quiet "${DOCKER_URL}"
    then
      IS_CACHED=1
    else
      echo "Could not find a cached container for the master branch either."
      IS_CACHED=0
    fi
  fi
  if [[ $IS_CACHED -eq 1 ]]
  then
    CACHE_FROM_CMD="--cache-from type=registry,ref=${DOCKER_URL}"
  fi
fi

# Run Docker build
set -x
python3 ops/docker_build.py \
  --container-def ${CONTAINER_DEF} \
  --container-id ${CONTAINER_ID} \
  ${BUILD_ARGS} \
  --cache-to type=inline \
  ${CACHE_FROM_CMD}
set +x

# Now cache the new container
if [[ ${USE_DOCKER_CACHE} -eq 1 ]]
then
    DOCKER_URL="${DOCKER_CACHE_REPO}/${CONTAINER_ID}:${DOCKER_TAG}"
    echo "docker tag ${CONTAINER_ID} ${DOCKER_URL}"
    docker tag "${CONTAINER_ID}" "${DOCKER_URL}"

    # Attempt to create Docker repository; it will fail if the repository already exists
    echo "aws ecr create-repository --repository-name ${CONTAINER_ID} --region ${DOCKER_CACHE_ECR_REGION}"
    if aws ecr create-repository --repository-name ${CONTAINER_ID} --region ${DOCKER_CACHE_ECR_REGION}
    then
      # Repository was created. Now set expiration policy
      echo "aws ecr put-lifecycle-policy --repository-name ${CONTAINER_ID}" \
           "--region ${DOCKER_CACHE_ECR_REGION} --lifecycle-policy-text file:///dev/stdin"
      echo "${ECR_LIFECYCLE_RULE}" | aws ecr put-lifecycle-policy --repository-name ${CONTAINER_ID} \
        --region ${DOCKER_CACHE_ECR_REGION} --lifecycle-policy-text file:///dev/stdin
    fi

    echo "docker push --quiet ${DOCKER_URL}"
    if ! time docker push --quiet "${DOCKER_URL}"
    then
        echo "ERROR: could not update Docker cache ${DOCKER_URL}"
        exit 1
    fi
fi
