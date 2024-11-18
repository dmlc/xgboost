#!/bin/bash

set -euo pipefail

source ops/pipeline/enforce-ci.sh

if [[ "$#" -lt 2 ]]
then
  echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64} {container_id}"
  exit 1
fi
  
suite="$1"
container_id="$2"

python3 ops/docker_run.py --container-id "${container_id}" \
  -- bash ops/pipeline/test-python-impl.sh "${suite}"
