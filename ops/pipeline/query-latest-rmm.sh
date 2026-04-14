#!/bin/bash
## Query latest version of RMM using GitHub CLI

set -euo pipefail

tmpfile="$(mktemp /tmp/abc-script.XXXXXX)"
cat >"$tmpfile" <<EOL
import fileinput
import re

from packaging.version import InvalidVersion, Version

versions = []
for e in fileinput.input():
    try:
        tag = e.strip()
        versions.append((tag, Version(tag)))
    except InvalidVersion:
        pass
latest_tag = max(versions, key=lambda x : x[1])[0]

m = re.search(r"v([0-9]{2}.[0-9]{2})", latest_tag)
print(m.group(1))
EOL
export RMM_VERSION=$(
  gh api repos/rapidsai/rmm/tags --paginate --jq '.[].name' | python3 "$tmpfile"
)
echo "--- Latest RMM version: ${RMM_VERSION}"
rm "$tmpfile"
