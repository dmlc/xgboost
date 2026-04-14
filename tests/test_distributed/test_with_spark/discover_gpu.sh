#!/usr/bin/env bash
set -euo pipefail

addresses=$(
  nvidia-smi --query-gpu=index --format=csv,noheader \
    | sed '/^[[:space:]]*$/d; s/^[[:space:]]*//; s/[[:space:]]*$//; s/.*/"&"/' \
    | paste -sd,
)
printf '{"name":"gpu","addresses":[%s]}\n' "${addresses}"
