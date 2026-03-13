#!/bin/bash
## Build docs for the R package and package it in a tarball.
##
## Prerequisites:
##   - BRANCH_NAME environment variable must be set
##   - R and required packages must be available
##
## This script uses ops/script/build_local_docs.py for the actual build.

set -euo pipefail

if [[ -z ${BRANCH_NAME:-} ]]; then
  echo "Error: BRANCH_NAME environment variable not set"
  exit 1
fi

# Setup R_LIBS_USER
export R_LIBS_USER="${R_LIBS_USER:-/tmp/rtmpdir}"
echo "R_LIBS_USER: ${R_LIBS_USER}"

mkdir -p "${R_LIBS_USER}"

# Container environment: fix permissions
if command -v gosu &>/dev/null; then
  gosu root chown -R "$UID:$GROUPS" "${R_LIBS_USER}"
fi

echo "--- Building R documentation ---"
set -x

python3 ops/script/build_local_docs.py r \
  --branch-name "${BRANCH_NAME}" \
  --r-libs-user "${R_LIBS_USER}"

# Verify output
[[ -f "r-docs-${BRANCH_NAME}.tar.bz2" ]] || {
  echo "Error: r-docs-${BRANCH_NAME}.tar.bz2 not found"
  exit 3
}
