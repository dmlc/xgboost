#!/bin/bash
## Query latest version of CCCL using GitHub CLI
## Note: RC version may be selected if available

set -euo pipefail

export CCCL_VERSION=$(python3 ops/pipeline/query-latest-version.py --repo NVIDIA/cccl)
echo "--- Latest CCCL version: ${CCCL_VERSION}"
