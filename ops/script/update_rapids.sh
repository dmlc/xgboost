#!/bin/bash

set -euo pipefail

LATEST_RAPIDS_VERSION=$(gh api repos/rapidsai/cuml/releases/latest --jq '.name' | sed -e 's/^v\([[:digit:]]\+\.[[:digit:]]\+\).*/\1/')
echo "LATEST_RAPIDS_VERSION = $LATEST_RAPIDS_VERSION"
DEV_RAPIDS_VERSION=$(date +%Y-%m-%d -d "20${LATEST_RAPIDS_VERSION//./-}-01 + 2 month" | cut -c3-7 | tr - .)
echo "DEV_RAPIDS_VERSION = $DEV_RAPIDS_VERSION"

OPS_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd -P )
CONTAINER_YAML="$OPS_PATH/docker/ci_container.yml"

sed -i "s/\&rapids_version \"[[:digit:]]\+\.[[:digit:]]\+\"/\&rapids_version \"${LATEST_RAPIDS_VERSION}\"/" \
  "$CONTAINER_YAML"
sed -i "s/\&dev_rapids_version \"[[:digit:]]\+\.[[:digit:]]\+\"/\&dev_rapids_version \"${DEV_RAPIDS_VERSION}\"/" \
  "$CONTAINER_YAML"
