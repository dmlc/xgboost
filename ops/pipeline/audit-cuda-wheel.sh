#!/bin/bash
## Audit XGBoost CUDA wheel for manylinux compliance
## This script runs inside the manylinux container (via GitHub Actions container support)

set -euo pipefail

if [[ -z "${GITHUB_SHA:-}" ]]
then
  echo "Make sure to set environment variable GITHUB_SHA"
  exit 1
fi

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {x86_64,aarch64} [--cuda-variant {cuda,cuda13}]"
  exit 2
fi
arch="$1"
cuda_variant="cuda"  # default

# Parse optional arguments
shift
while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda-variant)
      cuda_variant="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 2
      ;;
  esac
done

source ops/pipeline/classify-git-branch.sh

WHEEL_TAG=manylinux_2_28_${arch}

set -x

echo "--- Audit binary wheel to ensure it's compliant with ${WHEEL_TAG} standard"
raw_wheels=(python-package/dist/*.whl)
raw_wheel="${raw_wheels[0]}"

auditwheel repair --only-plat --plat ${WHEEL_TAG} "${raw_wheel}" --wheel-dir wheelhouse/
python3 -m wheel tags --python-tag cp312 --abi-tag abi3 --platform ${WHEEL_TAG} --remove \
  wheelhouse/*.whl
rm -v "${raw_wheel}"
mv -v wheelhouse/*.whl python-package/dist/

final_wheels=(python-package/dist/*.whl)
if [[ ${#final_wheels[@]} -ne 1 || ! -f "${final_wheels[0]}" ]]; then
  echo "error: expected exactly one final wheel in python-package/dist"
  exit 1
fi
final_wheel="${final_wheels[0]}"

if ! unzip -l "${final_wheel}" | grep libgomp > /dev/null; then
  echo "error: libgomp.so was not vendored in the wheel"
  exit -1
fi

# Check size of wheel
pydistcheck --config python-package/pyproject.toml "${final_wheel}"

# Generate meta.json only for the main CUDA variant (not cuda13)
if [[ $cuda_variant == "cuda" && $arch == "x86_64" ]]
then
  # Generate the meta info which includes xgboost version and the commit info
  # TODO(hcho3): Generate meta.json that contains both x86_64 and aarch64 wheels
  echo "--- Generate meta info"
  python3 ops/script/format_wheel_meta.py \
    --wheel-path "${final_wheel}"  \
    --commit-hash ${GITHUB_SHA}  \
    --platform-tag ${WHEEL_TAG}  \
    --meta-path python-package/dist/
fi

echo "--- Upload Python wheel"
if [[ ($is_pull_request == 0) && ($is_release_branch == 1) ]]
then
  python3 ops/pipeline/manage-artifacts.py upload \
    --s3-bucket xgboost-nightly-builds \
    --prefix ${BRANCH_NAME}/${GITHUB_SHA} --make-public \
    "${final_wheel}"

  if [[ $cuda_variant == "cuda" && $arch == "x86_64" ]]
  then
    python3 ops/pipeline/manage-artifacts.py upload \
      --s3-bucket xgboost-nightly-builds \
      --prefix ${BRANCH_NAME} --make-public \
      python-package/dist/meta.json
  fi
fi
