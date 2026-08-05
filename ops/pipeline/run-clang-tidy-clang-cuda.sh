#!/bin/bash
## Run clang-tidy using a clang-generated CUDA compilation database.

set -euo pipefail

build_dir="build-clang-tidy-cuda"
jobs=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      jobs="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1"
      echo "Usage: $0 [--jobs <n>]"
      exit 1
      ;;
  esac
done

if [[ -z "${jobs}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    jobs="$(sysctl -n hw.ncpu)"
  else
    jobs=4
  fi
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

"${repo_root}/ops/pipeline/build-cuda-clang.sh" \
  --build-dir "${repo_root}/${build_dir}" \
  --configure-only

if [[ -n "${XGBOOST_CLANG_PREFIX:-}" ]]; then
  clang_prefix="${XGBOOST_CLANG_PREFIX}"
elif command -v conda >/dev/null 2>&1; then
  clang_prefix="$(conda info --base)"
else
  if ! clang_path="$(command -v clang++)"; then
    echo "Could not find clang++ on PATH."
    echo "Install the clang toolchain, set XGBOOST_CLANG_PREFIX, or ensure clang++ is available on PATH."
    exit 1
  fi
  clang_prefix="$(cd "$(dirname "${clang_path}")/.." && pwd)"
fi

clang_run_tidy="${clang_prefix}/bin/run-clang-tidy"
if [[ ! -x "${clang_run_tidy}" ]]; then
  if ! clang_run_tidy="$(command -v run-clang-tidy)"; then
    echo "run-clang-tidy is required. Install conda-forge::clang-tools or set XGBOOST_CLANG_PREFIX to a clang prefix containing bin/run-clang-tidy."
    exit 1
  fi
fi

tidy_args=(
  -p "${repo_root}/${build_dir}" \
  -j "${jobs}" \
  -config-file "${repo_root}/.clang-tidy" \
  -header-filter "${repo_root}/(include|src)/.*" \
  -source-filter "${repo_root}/src/.*" \
  -extra-arg=-Wno-everything \
  -warnings-as-errors="*,-clang-diagnostic-*,-clang-analyzer-*" \
  -quiet
)

"${clang_run_tidy}" "${tidy_args[@]}"
