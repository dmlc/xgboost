#!/bin/bash

set -euo pipefail

source tests/buildkite/conftest.sh

echo "--- Test Python CPU ARM64"
buildkite-agent artifact download "python-package/dist/*.whl" . --step build-cpu-arm64
buildkite-agent artifact download "xgboost" . --step build-cpu-arm64
chmod +x ./xgboost
tests/ci_build/ci_build.sh aarch64 docker tests/ci_build/test_python.sh cpu-arm64
