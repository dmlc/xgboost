#!/bin/bash
## Query latest version of CCCL using GitHub CLI
## Note: RC version may be selected if available

set -euo pipefail

tmpfile="$(mktemp /tmp/abc-script.XXXXXX)"
cat >"$tmpfile" <<EOL
import fileinput

from packaging.version import InvalidVersion, Version

versions = []
for e in fileinput.input():
    try:
        tag = e.strip()
        versions.append((tag, Version(tag)))
    except InvalidVersion:
        pass
print(max(versions, key=lambda x : x[1])[0])
EOL
export CCCL_VERSION=$(
  gh api repos/NVIDIA/cccl/tags --paginate --jq '.[].name' | python3 "$tmpfile"
)
echo "--- Latest CCCL version: ${CCCL_VERSION}"
rm "$tmpfile"
