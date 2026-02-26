#!/bin/bash
## Build docs for the JVM packages and package it in a tarball.
##
## Prerequisites:
##   - libxgboost4j.so must be built and placed in lib/
##   - BRANCH_NAME environment variable must be set
##
## This script uses ops/script/build_local_docs.py for the actual build.

set -euo pipefail

if [[ -z ${BRANCH_NAME:-} ]]; then
  echo "Error: BRANCH_NAME environment variable not set"
  exit 1
fi

if [[ ! -f lib/libxgboost4j.so ]]; then
  echo "Error: lib/libxgboost4j.so not found. Build it first with:"
  echo "  python3 ops/script/build_local_docs.py jvm-lib"
  exit 2
fi

echo "--- Building JVM documentation ---"
set -x

python3 ops/script/build_local_docs.py jvm --branch-name "${BRANCH_NAME}"

# Verify output
[[ -f "jvm-packages/${BRANCH_NAME}.tar.bz2" ]] || {
  echo "Error: jvm-packages/${BRANCH_NAME}.tar.bz2 not found"
  exit 3
}
