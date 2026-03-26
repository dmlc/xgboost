#!/bin/bash
## Query latest version of RMM using GitHub CLI

set -euo pipefail

export RMM_VERSION=$(python3 ops/pipeline/query-latest-version.py --repo rapidsai/rmm --extract-minor)
echo "--- Latest RMM version: ${RMM_VERSION}"
