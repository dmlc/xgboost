#!/bin/bash
# Estimate the opportunity for preloading Docker images into the AWS runner image.
# This same-runner comparison does not measure AMI boot time or snapshot lazy reads.
# Remove this experiment after deciding whether a separate preloaded-AMI trial is useful.
set -euo pipefail

source ops/pipeline/get-docker-registry-details.sh
source ops/pipeline/get-image-tag.sh

result_dir="${PWD}/container-startup-results"
mkdir -p "$result_dir"
repository=xgb-ci.gpu
# Resolve the mutable tag before either pull so both passes use identical layers.
digest=$(aws ecr describe-images --region "$ECR_AWS_REGION" \
  --registry-id "$ECR_AWS_ACCOUNT_ID" --repository-name "$repository" \
  --image-ids "imageTag=${IMAGE_TAG}" --query 'imageDetails[0].imageDigest' --output text)
if [[ "$digest" != sha256:* ]]; then
  echo "Could not resolve ${repository}:${IMAGE_TAG}" >&2
  exit 1
fi
image="${DOCKER_REGISTRY_URL}/${repository}@${digest}"
{
  echo "Image: ${image}"
  echo "Source tag: ${IMAGE_TAG}"
  echo "Commit: $(git rev-parse HEAD)"
  date -u
  uname -a
  docker version
  docker info
  nvidia-smi
  df -h
} > "$result_dir/environment.txt" 2>&1
docker image ls --digests --no-trunc > "$result_dir/images-before.txt"
docker system df -v > "$result_dir/storage-before.txt"

# Do not prune the runner. Existing/shared layers can make the initial pull partly
# warm; preserve the inventory rather than claiming a guaranteed cold baseline.
printf 'pass\toperation\tseconds\n' > "$result_dir/timings.tsv"
measure() {
  local pass=$1 operation=$2
  shift 2
  /usr/bin/time -f "${pass}\t${operation}\t%e" -a -o "$result_dir/timings.tsv" \
    "$@" 2>&1 | tee "$result_dir/${pass}-${operation}.log"
}

# Use a fresh container for every invocation. No CuPy cache is shared between
# passes: the intended variable is Docker's local image/layer cache only.
smoke_test=$(cat <<'PY'
import cupy as cp

assert cp.cuda.runtime.getDeviceCount() > 0
kernel = cp.RawKernel(
    'extern "C" __global__ void fill(float* x) { x[threadIdx.x] = 1.0f; }',
    'fill',
)
x = cp.empty(32, dtype=cp.float32)
kernel((1,), (32,), (x,))
cp.cuda.Stream.null.synchronize()
assert (cp.asnumpy(x) == 1.0).all()
print("GPU compilation, execution and host transfer passed")
PY
)
for pass in initial cached; do
  measure "$pass" pull docker pull "$image"
  measure "$pass" startup docker run --rm --pull=never --gpus all \
    --entrypoint /bin/true "$image"
  # Includes fresh-container startup, Python/CuPy initialization and GPU work.
  # This is an environment smoke test, not the full XGBoost test suite.
  measure "$pass" gpu-smoke docker run --rm --pull=never --gpus all \
    --entrypoint /bin/bash "$image" -ec \
    'source activate gpu_test; exec python -c "$1"' bash "$smoke_test"
done

docker system df -v > "$result_dir/storage-after.txt"
{
  echo '### GPU container startup measurements'
  echo
  echo "Image: \`${image}\`"
  echo
  echo '| Pass | Operation | Seconds |'
  echo '| --- | --- | ---: |'
  awk -F '\t' 'NR > 1 { printf "| %s | %s | %s |\n", $1, $2, $3 }' "$result_dir/timings.tsv"
  echo
  echo 'Pull time includes downloading and extracting layers; these are not timed separately.'
  echo 'Initial pull may reuse pre-existing layers; see the image and storage inventories.'
  echo 'Each startup and GPU smoke measurement uses a separate fresh container.'
  echo 'GPU smoke includes container startup; do not add it to the standalone startup measurement.'
  echo 'This comparison excludes runner provisioning and does not establish AMI-preloading savings.'
} | tee "$result_dir/summary.md" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
