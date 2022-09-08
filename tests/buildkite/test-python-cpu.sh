#!/bin/bash

set -euo pipefail

echo "--- Test CPU code in Python env"

source tests/buildkite/conftest.sh

mkdir -pv python-package/dist
buildkite-agent artifact download "python-package/dist/*.whl" . --step build-cuda
buildkite-agent artifact download "xgboost" . --step build-cpu
chmod +x ./xgboost

tests/ci_build/ci_build.sh cpu docker tests/ci_build/test_python.sh cpu
