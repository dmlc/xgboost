#!/bin/bash
## Run clang-tidy using a clang-generated CUDA compilation database.

set -euo pipefail

clang_version="21.1.8"
cmake_version="4.2.3"
build_dir="build-clang-tidy-cuda"
jobs="${XGBOOST_TIDY_JOBS:-}"
checks="${XGBOOST_TIDY_CHECKS:-}"
extra_args_csv="${XGBOOST_TIDY_EXTRA_ARGS:--Wno-everything}"
files_csv="${XGBOOST_TIDY_FILES:-}"
source_filter="${XGBOOST_TIDY_SOURCE_FILTER:-}"
warnings_as_errors="${XGBOOST_TIDY_WARNINGS_AS_ERRORS:-*,-clang-diagnostic-*,-clang-analyzer-*}"

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
    --extra-arg)
      extra_args_csv="$2"
      shift 2
      ;;
    --files)
      files_csv="$2"
      shift 2
      ;;
    --source-filter)
      source_filter="$2"
      shift 2
      ;;
    --warnings-as-errors)
      warnings_as_errors="$2"
      shift 2
      ;;
    *)
      echo "Unrecognized argument: $1"
      echo "Usage: $0 [--clang-version <version>] [--cmake-version <version>] [--build-dir <dir>] [--jobs <n>] [--checks <filter>] [--extra-arg <comma-separated-extra-args>] [--files <comma-separated-files>] [--source-filter <regex>] [--warnings-as-errors <filter>]"
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
if [[ -z "${source_filter}" ]]; then
  source_filter="${repo_root}/src/.*"
fi

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
    echo "run-clang-tidy is required. Install clang-tools or set XGBOOST_CLANG_PREFIX to a clang prefix containing bin/run-clang-tidy."
    exit 1
  fi
fi

IFS=',' read -r -a extra_args <<< "${extra_args_csv}"
IFS=',' read -r -a tidy_files <<< "${files_csv}"

tidy_args=(
  -p "${repo_root}/${build_dir}" \
  -j "${jobs}" \
  -config-file "${repo_root}/.clang-tidy" \
  -header-filter "${repo_root}/(include|src)/.*" \
  -source-filter "${source_filter}" \
  -quiet
)

if [[ -n "${checks}" ]]; then
  tidy_args+=(-checks="${checks}")
fi
if [[ -n "${extra_args_csv}" ]]; then
  for extra_arg in "${extra_args[@]}"; do
    tidy_args+=(-extra-arg="${extra_arg}")
  done
fi
if [[ -n "${warnings_as_errors}" ]]; then
  tidy_args+=(-warnings-as-errors="${warnings_as_errors}")
fi
if [[ -n "${files_csv}" ]]; then
  tidy_args+=("${tidy_files[@]}")
fi

"${clang_run_tidy}" "${tidy_args[@]}"
