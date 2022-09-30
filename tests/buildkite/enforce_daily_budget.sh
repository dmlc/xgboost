#!/bin/bash

set -euo pipefail

echo "--- Enforce daily budget"

source tests/buildkite/conftest.sh

if [[ $is_release_branch == 1 ]]
then
  echo "Automatically approving all test jobs for release branches"
else
  python3 tests/buildkite/enforce_daily_budget.py
fi
