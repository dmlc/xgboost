#!/bin/bash

set -euo pipefail

LATEST_RAPIDS_VERSION=$(gh api repos/rapidsai/cuml/releases/latest --jq '.name' | sed -e 's/^v\([[:digit:]]\+\.[[:digit:]]\+\).*/\1/')
echo "LATEST_RAPIDS_VERSION = $LATEST_RAPIDS_VERSION"

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

sed -i "s/^RAPIDS_VERSION=[[:digit:]]\+\.[[:digit:]]\+/RAPIDS_VERSION=${LATEST_RAPIDS_VERSION}/" $PARENT_PATH/conftest.sh
