#!/bin/bash

set -euo pipefail

cmake_files=$(
    find . -name CMakeLists.txt -o -path "./cmake/*.cmake" \
    | grep -v dmlc-core \
    | grep -v gputreeshap
)
cmakelint \
    --linelength=120 \
    --filter=-convention/filename,-package/stdargs,-readability/wonkycase \
    ${cmake_files} \
|| exit 1
