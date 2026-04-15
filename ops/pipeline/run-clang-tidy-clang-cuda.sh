#!/bin/bash
## Smoke-test clang-tidy using a clang-generated CUDA compilation database.

set -euo pipefail

clang_version="21.1.8"
cmake_version="4.2.3"
build_dir="build-clang-tidy-cuda"
jobs="${XGBOOST_TIDY_JOBS:-}"
checks="${XGBOOST_TIDY_CHECKS:--*,google-runtime-int}"
files_csv="${XGBOOST_TIDY_FILES:-src/common/timer.cc,src/predictor/interpretability/shap.cu}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clang-version)
      clang_version="$2"
      shift 2
      ;;
    --cmake-version)
      cmake_version="$2"
      shift 2
      ;;
    --build-dir)
      build_dir="$2"
      shift 2
      ;;
    --jobs)
      jobs="$2"
      shift 2
      ;;
    --checks)
      checks="$2"
      shift 2
      ;;
    --files)
      files_csv="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1"
      echo "Usage: $0 [--clang-version <version>] [--cmake-version <version>] [--build-dir <dir>] [--jobs <n>] [--checks <filter>] [--files <comma-separated-files>]"
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
  if (( jobs > 35 )); then
    jobs=35
  fi
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

"${repo_root}/ops/pipeline/build-cuda-clang.sh" \
  --clang-version "${clang_version}" \
  --cmake-version "${cmake_version}" \
  --build-dir "${repo_root}/${build_dir}" \
  --configure-only

if [[ -n "${XGBOOST_CLANG_PREFIX:-}" ]]; then
  clang_prefix="${XGBOOST_CLANG_PREFIX}"
elif command -v conda >/dev/null 2>&1; then
  clang_prefix="$(conda info --base)"
else
  clang_path="$(command -v clang++)"
  clang_prefix="$(cd "$(dirname "${clang_path}")/.." && pwd)"
fi

clang_run_tidy="${clang_prefix}/bin/run-clang-tidy"
if [[ ! -x "${clang_run_tidy}" ]]; then
  clang_run_tidy="$(command -v run-clang-tidy)"
fi

IFS=',' read -r -a tidy_files <<< "${files_csv}"

"${clang_run_tidy}" \
  -p "${repo_root}/${build_dir}" \
  -j "${jobs}" \
  -checks="${checks}" \
  -quiet \
  "${tidy_files[@]}"
