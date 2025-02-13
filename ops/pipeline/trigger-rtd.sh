#!/bin/bash
## Trigger a new build on ReadTheDocs service.

set -euo pipefail

if [[ -z ${BRANCH_NAME:-} ]]
then
  echo "Make sure to define environment variable BRANCH_NAME."
  exit 1
fi

echo "Branch name: ${BRANCH_NAME}"
export RTD_AUTH_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id runs-on/readthedocs-auth-token --output text \
  --region us-west-2 --query SecretString || echo -n '')
python3 ops/pipeline/trigger-rtd-impl.py
