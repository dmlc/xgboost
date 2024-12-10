## Log into AWS ECR (Elastic Container Registry) to be able to pull containers from it
## Note. Requires valid AWS credentials

set -euo pipefail

source ops/pipeline/get-docker-registry-details.sh

echo "aws ecr get-login-password --region ${ECR_AWS_REGION} |" \
    "docker login --username AWS --password-stdin ${DOCKER_REGISTRY_URL}"
aws ecr get-login-password --region ${ECR_AWS_REGION} \
  | docker login --username AWS --password-stdin ${DOCKER_REGISTRY_URL}
